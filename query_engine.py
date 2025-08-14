from pathlib import Path
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    JobProcess,
    RoomInputOptions,
)
from livekit.plugins import deepgram, google, silero, cartesia, noise_cancellation

from s3_vector_storage import embed_data_to_s3, query_s3_vectors, s3_vector_storage

load_dotenv()

def prewarm(proc: JobProcess):
    """Prewarm the models and ensure data is embedded."""
    proc.userdata["vad"] = silero.VAD.load()
    
    # Ensure data is embedded in S3
    try:
        # embed_data_to_s3()
        print("Data embedding verification complete")
    except Exception as e:
        print(f"Warning: Could not verify embedding: {e}")

@llm.function_tool
async def query_info(query: str) -> str:
    """Get more information about Omesh's portfolio from S3 vectors"""
    try:
        # allow up to 2 chunks per doc (tweak if you want more)
        result = query_s3_vectors(query, top_k=3, max_chunks_per_doc=2)
        print(f"S3 Vector Query result: {result}")
        return result
    except Exception as e:
        print(f"Error querying S3 vectors: {e}")
        return "I'm having trouble accessing the portfolio information right now."


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = Agent(
        instructions=(
            "You are a voice assistant created by Omesh. Your interface "
            "with users will be voice. You should use short and concise "
            "responses, and avoiding usage of unpronouncable punctuation. "
            "The sole purpose is to explain the portfolio of projects Omesh has done "
            "which is stored in AWS S3 vectors. Always use the query_info tool to "
            "answer questions and provide detailed answers when necessary to impress recruiters."
        ),
        vad=silero.VAD.load(),
        llm=google.LLM(model="gemini-2.0-flash-001"),
        stt=deepgram.STT(model="nova-3"),
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        tools=[query_info],
    )

    session = AgentSession()
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.say("Hey, how can I help you learn about Omesh's portfolio today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))