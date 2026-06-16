import os
import sys
from pathlib import Path

from gtts import gTTS
from langchain_core.prompts import PromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.chatbot import CleanHuggingFacePipeline, clean_response

STORY_PROMPT = PromptTemplate(
    input_variables=["topic"],
    template=(
        "<|im_start|>user\n"
        "Write a short engaging educational story about {topic} for beginners. "
        "Use simple and clear language. Keep it around 100-150 words. "
        "End with one brief summary sentence. "
        "Write only the story text with no headings or role labels.\n"
        "<|im_start|>assistant\n"
    ),
)

OUTPUT_DIR = Path(__file__).parent / "output"


def create_story_llm():
    """Create SmolLM2 with a higher token limit for story generation."""
    return CleanHuggingFacePipeline.from_model_id(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 300,
            "temperature": 0.3,
            "return_full_text": False,
        },
    )


def generate_story(topic: str, llm=None) -> str:
    """Generate speakable text using the local SmolLM2 model."""
    llm = llm or create_story_llm()
    chain = STORY_PROMPT | llm
    response = chain.invoke({"topic": topic})
    story = clean_response(str(response))
    if not story.strip():
        raise ValueError(
            "SmolLM2 returned empty text. "
            "Try running again or use a simpler topic."
        )
    return story


def text_to_speech(text: str, output_path: Path) -> Path:
    """Convert text to an MP3 file with gTTS."""
    tts = gTTS(text=text, lang="en")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts.save(str(output_path))
    return output_path


def speak_topic(topic: str) -> None:
    """Generate a story with SmolLM2, convert it to speech, and play it."""
    print(f"Loading SmolLM2 and generating story about: {topic}")
    llm = create_story_llm()
    story = generate_story(topic, llm)

    print("\nGenerated story:\n")
    print(story)

    output_path = OUTPUT_DIR / "story.mp3"
    print("\nConverting to speech...")
    text_to_speech(story, output_path)
    print(f"Saved audio to {output_path}")
    os.startfile(output_path)


if __name__ == "__main__":
    topic = "the life cycle of butterflies"
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    speak_topic(topic)
