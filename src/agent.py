import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional, List
import asyncio
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import deepgram, cartesia, groq, silero, noise_cancellation

# ----------------------------------------------------
# Setup
# ----------------------------------------------------
logger = logging.getLogger("interview-prep-agent")
logger.setLevel(logging.INFO)
load_dotenv(".env.local")

# ----------------------------------------------------
# User Data Storage (per-call session)
# ----------------------------------------------------
@dataclass
class JobSeekerData:
    name: Optional[str] = None
    age: Optional[str] = None
    location: Optional[str] = None
    job_interest: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    current_stage: str = "greeting"  # greeting -> info_collection -> concept_explanation -> skill_assessment -> practice_intro -> final_qna -> wrap_up
    conversation_start_time: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    skill_responses: List[str] = field(default_factory=list)
    practice_intro_done: bool = False

    def is_basic_info_complete(self) -> bool:
        """Check if basic info collection is complete"""
        return all([self.name, self.age, self.location, self.job_interest])
    
    def get_job_specific_skills(self) -> List[str]:
        """Get relevant skills based on job interest"""
        job_skills = {
            "delivery agent": ["punctuality", "customer service", "navigation", "phone handling", "vehicle safety"],
            "plumber": ["pipe fitting", "leak detection", "tool usage", "customer communication", "emergency handling"],
            "electrician": ["wiring", "safety protocols", "multimeter usage", "troubleshooting", "electrical codes"],
            "mechanic": ["engine diagnosis", "tool handling", "problem solving", "customer explanation", "safety practices"],
            "healthcare worker": ["patient care", "hygiene", "empathy", "following protocols", "emergency response"],
            "it support": ["computer basics", "troubleshooting", "customer patience", "problem solving", "technical communication"]
        }
        if self.job_interest and self.job_interest.lower() in job_skills:
            return job_skills[self.job_interest.lower()]
        return ["communication", "punctuality", "problem solving", "teamwork", "reliability"]

# Type alias for RunContext with our JobSeekerData
RunContext_T = RunContext[JobSeekerData]

# ----------------------------------------------------
# Tool functions to update user data
# ----------------------------------------------------
@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The job seeker's name")],
    context: RunContext_T,
) -> str:
    context.userdata.name = name.strip()
    return f"Got it, your name is {name}"

@function_tool()
async def update_age(
    age: Annotated[str, Field(description="The job seeker's age")],
    context: RunContext_T,
) -> str:
    context.userdata.age = age.strip()
    return f"Your age is {age}"

@function_tool()
async def update_location(
    location: Annotated[str, Field(description="City and state where the job seeker lives")],
    context: RunContext_T,
) -> str:
    context.userdata.location = location.strip()
    return f"You are from {location}"

@function_tool()
async def update_job_interest(
    job: Annotated[str, Field(description="The type of job they are looking for (plumber, electrician, delivery agent, mechanic, healthcare worker, IT support, etc)")],
    context: RunContext_T,
) -> str:
    context.userdata.job_interest = job.strip()
    return f"You are looking for {job} job"

@function_tool()
async def update_skills(
    skills: Annotated[List[str], Field(description="Skills the person has mentioned they know")],
    context: RunContext_T,
) -> str:
    context.userdata.skills.extend(skills)
    skill_str = ", ".join(skills)
    return f"Got it, you know {skill_str}"

@function_tool()
async def update_challenges(
    challenges: Annotated[List[str], Field(description="Challenges or issues the person faces in their job search")],
    context: RunContext_T,
) -> str:
    context.userdata.challenges.extend(challenges)
    return "I understand the challenges you are facing"

@function_tool()
async def record_skill_response(
    response: Annotated[str, Field(description="User's response to a skill-based question")],
    context: RunContext_T,
) -> str:
    context.userdata.skill_responses.append(response)
    return "I have noted your response"

@function_tool()
async def move_to_concept_explanation(
    context: RunContext_T,
) -> str:
    """Move to explaining interview concepts"""
    context.userdata.current_stage = "concept_explanation"
    return "Moving to explain interview concepts"

@function_tool()
async def move_to_skill_assessment(
    context: RunContext_T,
) -> str:
    """Move to skill assessment phase"""
    context.userdata.current_stage = "skill_assessment"
    return "Starting skill assessment"

@function_tool()
async def move_to_practice_intro(
    context: RunContext_T,
) -> str:
    """Move to practice introduction phase"""
    context.userdata.current_stage = "practice_intro"
    return "Moving to practice introduction"

@function_tool()
async def move_to_final_qna(
    context: RunContext_T,
) -> str:
    """Move to final Q&A phase"""
    context.userdata.current_stage = "final_qna"
    return "Moving to final questions"

@function_tool()
async def move_to_wrap_up(
    context: RunContext_T,
) -> str:
    """Move to wrap up and final feedback"""
    context.userdata.current_stage = "wrap_up"
    return "Moving to final feedback and wrap up"

@function_tool()
async def mark_practice_intro_done(
    context: RunContext_T,
) -> str:
    """Mark that practice introduction is complete"""
    context.userdata.practice_intro_done = True
    return "Practice introduction completed"

@function_tool()
async def start_conversation_timer(
    context: RunContext_T,
) -> str:
    """Start tracking conversation time"""
    import time
    context.userdata.conversation_start_time = time.time()
    return "Timer started"

@function_tool()
async def update_languages(
    languages: Annotated[List[str], Field(description="Languages the person speaks")],
    context: RunContext_T,
) -> str:
    context.userdata.languages = languages
    lang_str = ", ".join(languages)
    return f"You speak {lang_str}"

@function_tool()
async def move_to_setup_phase(
    context: RunContext_T,
) -> str:
    """Move to interview setup explanation phase"""
    context.userdata.current_stage = "concept_explanation"
    return "Moving to interview preparation concepts"

@function_tool()
async def move_to_practice_phase(
    context: RunContext_T,
) -> str:
    """Move to practice phase"""
    context.userdata.current_stage = "practice_intro"
    return "Starting interview practice"

@function_tool()
async def add_note(
    note: Annotated[str, Field(description="Add a note about the conversation")],
    context: RunContext_T,
) -> str:
    context.userdata.notes.append(note)
    return "Note added"

# ----------------------------------------------------
# Main Interview Prep Agent
# ----------------------------------------------------
class InterviewPrepAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly interview preparation coach for blue and gray collar workers in India. "
                "Your goal is to help job seekers understand interviews, build confidence, and become interview-ready. "
                
                "PERSONALITY AND TONE: "
                "- Speak like a supportive mentor who genuinely cares about their success "
                "- Use simple, clear language that anyone can understand "
                "- Be patient, encouraging, and reassuring throughout "
                "- Make them feel comfortable and confident "
                "- Use Hinglish phrases naturally when appropriate "
                
                "CONVERSATION WORKFLOW (IMPORTANT - Follow this sequence): "
                
                "1. GREETING PHASE: "
                "- Warmly welcome them and introduce yourself as their interview coach "
                "- Explain that you will help them prepare for job interviews and make them interview-ready "
                "- Make them feel comfortable and start conversation timer "
                
                "2. INFO COLLECTION PHASE (Keep this focused): "
                "- Collect: Name, Age, City/Location "
                "- What job they are looking for (delivery agent, plumber, electrician, IT support, healthcare worker, mechanic) "
                "- What languages they speak (English, Hindi, Kannada, etc.) "
                "- Ask about challenges they face in job searching "
                "- Ask what skills they already have related to their target job "
                
                "3. CONCEPT EXPLANATION PHASE (3-5 minutes): "
                "- Explain what an interview is in simple terms "
                "- Explain what a resume is and why it matters "
                "- Talk about soft skills (punctuality, communication, teamwork) "
                "- Explain what employers look for in their specific job type "
                "- Ask if they understand before moving forward "
                
                "4. SKILL ASSESSMENT PHASE: "
                "- Ask them specific situation-based questions related to their target job "
                "- For plumber: 'What would you do if a customer's pipe is leaking badly?' "
                "- For electrician: 'How do you stay safe when working with electricity?' "
                "- For delivery agent: 'What if a customer is angry about late delivery?' "
                "- For IT support: 'How do you help someone who says their computer is not working?' "
                "- For healthcare worker: 'How do you make a patient feel comfortable?' "
                "- Listen to their response and give INSTANT feedback on what was good and what they could improve "
                
                "5. PRACTICE INTRODUCTION PHASE: "
                "- Summarize what you learned about them (skills, background, job interest) "
                "- Ask them to introduce themselves as if they are in a real interview "
                "- Listen and judge their response "
                "- Give feedback on what they missed or could improve "
                "- Have a small discussion about specific job requirements "
                
                "6. FINAL Q&A PHASE: "
                "- Ask if they want to practice anything specific again "
                "- Answer any questions they have about interviews or their target job "
                "- Ask if they need further explanation on any topic "
                
                "7. WRAP UP PHASE: "
                "- Provide final comprehensive feedback covering: "
                "  * Key interview tips for their specific job "
                "  * Skills they should highlight to employers "
                "  * Soft skills they should demonstrate "
                "  * Areas they need to work on "
                "- End on an encouraging note "
                
                "TARGET: Complete everything in about 10 minutes maximum "
                
                "IMPORTANT FORMATTING RULES: "
                "- NEVER use markdown, asterisks, bold, italics, or any formatting symbols "
                "- NEVER use numbered lists (1. 2. 3.) or bullet points (- or *) "
                "- NEVER use headings with # symbols "
                "- Speak in natural, conversational sentences only "
                "- Use words like 'first', 'then', 'also', 'another thing' instead of lists "
                "- Keep responses conversational and natural for text-to-speech "
                
                "JOB-SPECIFIC INTERVIEW FOCUS: "
                "- Delivery Agent: punctuality, customer service, phone handling, navigation, vehicle safety "
                "- Electrician: safety protocols, basic electrical knowledge, tool familiarity, problem-solving "
                "- Plumber: problem-solving, customer interaction, tool knowledge, emergency handling, reliability "
                "- Mechanic: technical skills, diagnostic abilities, customer communication, safety practices "
                "- Healthcare Worker: patient care, hygiene, empathy, following protocols, compassion "
                "- IT Support: basic computer knowledge, problem-solving, patience with customers, clear communication "
                
                "Remember: These are hardworking people who may not understand interview processes. Be patient and supportive."
            ),
            tools=[
                update_name, update_age, update_location, update_job_interest,
                update_languages, update_skills, update_challenges, record_skill_response,
                move_to_concept_explanation, move_to_skill_assessment, move_to_practice_intro,
                move_to_final_qna, move_to_wrap_up, mark_practice_intro_done,
                start_conversation_timer, add_note
            ],
            llm=groq.LLM(model="llama-3.1-8b-instant", temperature=0.4),
            tts=cartesia.TTS(voice="95856005-0332-41b0-935f-352e296aa0df"),  # Warm conversational voice
        )

    async def on_enter(self) -> None:
        """Called when the agent starts"""
        logger.info("Interview Prep Agent starting")
        
        userdata: JobSeekerData = self.session.userdata
        
        # Initialize conversation timer if not set
        if userdata.conversation_start_time is None:
            import time
            userdata.conversation_start_time = time.time()
        
        # Calculate elapsed time
        elapsed_time = 0
        if userdata.conversation_start_time:
            import time
            elapsed_time = (time.time() - userdata.conversation_start_time) / 60  # in minutes
        
        # Add context about current conversation state
        context_msg = (
            f"Current conversation stage: {userdata.current_stage}\n"
            f"Elapsed time: {elapsed_time:.1f} minutes (target: ~10 minutes total)\n"
            f"User info collected:\n"
            f"Name: {userdata.name or 'Not provided'}\n"
            f"Age: {userdata.age or 'Not provided'}\n"
            f"Location: {userdata.location or 'Not provided'}\n"
            f"Job Interest: {userdata.job_interest or 'Not provided'}\n"
            f"Languages: {', '.join(userdata.languages) if userdata.languages else 'Not provided'}\n"
            f"Skills mentioned: {', '.join(userdata.skills) if userdata.skills else 'None yet'}\n"
            f"Challenges: {', '.join(userdata.challenges) if userdata.challenges else 'None mentioned'}\n"
            f"Skill responses recorded: {len(userdata.skill_responses)}\n"
            f"Practice intro done: {userdata.practice_intro_done}\n"
            f"Job-specific skills to assess: {', '.join(userdata.get_job_specific_skills())}\n"
            f"Notes: {'; '.join(userdata.notes) if userdata.notes else 'None'}\n\n"
            
            "STAGE-SPECIFIC GUIDANCE:\n"
            f"- If greeting stage: Give warm welcome, explain you'll help with interview prep, start timer\n"
            f"- If info_collection stage: Ask for missing basic info (name, age, location, job interest, languages, skills, challenges)\n"
            f"- If concept_explanation stage: Explain interviews, resumes, soft skills, job-specific expectations (3-5 min)\n"
            f"- If skill_assessment stage: Ask situation-based questions for their job, give instant feedback\n"
            f"- If practice_intro stage: Have them introduce themselves, give feedback, discuss job requirements\n"
            f"- If final_qna stage: Ask what they want to practice more, answer questions\n"
            f"- If wrap_up stage: Give comprehensive final feedback and encouragement\n\n"
            
            "Move to next stage when current objectives are complete. "
            "Keep responses natural and conversational without any formatting. "
            "Be supportive and encouraging throughout."
        )
        
        # Update chat context with current state
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(role="system", content=context_msg)
        await self.update_chat_ctx(chat_ctx)

        # Generate initial response
        self.session.generate_reply(tool_choice="auto")

# ----------------------------------------------------
# Entrypoint function
# ----------------------------------------------------
async def entrypoint(ctx: JobContext):
    """Main entry point for the LiveKit agent"""
    logger.info("Starting Interview Prep Agent")
    
    # Pre-warm VAD for faster voice activity detection
    vad = silero.VAD.load()
    
    # Create user data storage for this session
    userdata = JobSeekerData()
    
    # Create the agent session
    session = AgentSession[JobSeekerData](
        userdata=userdata,
        stt=deepgram.STT(
            model="nova-2", 
            language="en-IN"  # Indian English for better accent recognition
        ),
        llm=groq.LLM(model="llama-3.1-8b-instant", temperature=0.4),
        tts=cartesia.TTS(voice="95d51f79-c397-46f9-b49a-23763d3eaa2d", language="en"),  # Warm conversational voice
        vad=vad,
        max_tool_steps=3,  # Limit tool calls to prevent long delays
        preemptive_generation=True,  # Generate responses while user is speaking for smoother flow
        
        # Voice activity settings for better user experience
        min_endpointing_delay=0.8,  # Wait 0.8s of silence before responding
        max_endpointing_delay=2.0,  # Allow up to 2s for continued speech
        allow_interruptions=True,   # Users can interrupt the agent
    )

    # Start the session
    await session.start(
        agent=InterviewPrepAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),  # Background noise cancellation
        ),
    )
    
    logger.info("Interview Prep Agent session started successfully")

# ----------------------------------------------------
# Main execution
# ----------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))