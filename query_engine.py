from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
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
from livekit.plugins import deepgram, google, silero, cartesia

load_dotenv()

# check if storage already exists
THIS_DIR = Path(__file__).parent
PERSIST_DIR = THIS_DIR / "query-engine-storage"

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = GoogleGenAI(
    model="gemini-2.0-flash",
)

if not PERSIST_DIR.exists():
    # load the documents and create the index
    documents = SimpleDirectoryReader(THIS_DIR / "data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


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
        tools=[query_info],
    )

    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

    await session.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
