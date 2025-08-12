from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)

from livekit.agents import BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import deepgram, google, silero, cartesia, noise_cancellation

from embed_query_engine import needs_reembedding, embed_data

load_dotenv()

# Directories
THIS_DIR = Path(__file__).parent
PERSIST_DIR = THIS_DIR / "query-engine-storage"

# Configure models
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = GoogleGenAI(model="gemini-2.0-flash")

# Check if embedding is needed and load/create index
# if needs_reembedding():
#     embed_data()

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


@llm.function_tool
async def query_info(query: str) -> str:
    """Get more information about a specific topic"""
    query_engine = index.as_query_engine()
    res = await query_engine.aquery(query)
    print("Query result:", res)
    return str(res)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = Agent(
        instructions=(
            "You are a voice assistant created by Omesh. Your interface "
            "with users will be voice. You should use short and concise "
            "responses, and avoiding usage of unpronouncable punctuation."
            "The sole purpose is explain the portfolio of project omesh done which is summarized in knowledge base, always use that to answer and impress the recuireter and give little detailed answers when necessary "
        ),
        vad=silero.VAD.load(),
        llm=google.LLM(model="gemini-2.0-flash-001"),
        stt=deepgram.STT(model="nova-3"),
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        # tts=google.TTS(
        #     gender="female",
        #     voice_name="en-US-Standard-H",
        # ),
        tools=[query_info],
    )

    session = AgentSession()
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
