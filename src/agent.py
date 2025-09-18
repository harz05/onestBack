import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Annotated
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import (
    JobContext, 
    WorkerOptions, 
    cli, 
    Agent,
    AgentSession,
    RunContext
)
from livekit.agents.llm import function_tool, ChatContext, ChatMessage, LLM
from livekit.plugins import groq, silero, deepgram, cartesia, sarvam
from pydantic import Field

# Load environment variables  
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env.local')

logger = logging.getLogger("interview-prep-agent")
logger.setLevel(logging.INFO)

# ----------------------------------------------------
# User Data Schema
# ----------------------------------------------------
@dataclass
class JobSeekerProfile:
    name: Optional[str] = None
    age: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    target_job: Optional[str] = None  # delivery agent, plumber, electrician, etc.
    current_status: Optional[str] = None  # applied, looking, preparing
    languages: List[str] = field(default_factory=list)  # [English, Kannada, etc.]
    experience_level: Optional[str] = None  # beginner, some experience, experienced
    work_schedule: Optional[str] = None
    main_challenges: List[str] = field(default_factory=list)
    
    # Session state
    current_phase: str = "introduction"  # introduction -> setup -> work_analysis -> teaching -> practice -> feedback
    concepts_understood: bool = False
    ready_for_practice: bool = False
    interview_completed: bool = False
    
    # Job-specific knowledge
    job_skills: Dict[str, List[str]] = field(default_factory=dict)
    safety_points: List[str] = field(default_factory=list)
    common_questions: List[str] = field(default_factory=list)

# Type alias for RunContext with JobSeekerProfile
RunContext_T = RunContext[JobSeekerProfile]

# ----------------------------------------------------
# Job-specific knowledge base
# ----------------------------------------------------
JOB_KNOWLEDGE = {
    "delivery_agent": {
        "skills": ["Time management", "Route optimization", "Customer service", "Package handling"],
        "safety": ["Traffic rules", "Proper lifting techniques", "Weather precautions", "Vehicle maintenance"],
        "common_questions": [
            "How do you handle difficult customers?",
            "What would you do if a package is damaged?",
            "How do you manage time during peak hours?",
            "Describe your experience with navigation apps"
        ],
        "daily_tasks": ["Package pickup", "Route planning", "Customer interaction", "Vehicle checks"]
    },
    "electrician": {
        "skills": ["Circuit analysis", "Safety protocols", "Tool usage", "Problem diagnosis"],
        "safety": ["Electrical safety", "PPE usage", "Circuit isolation", "Emergency procedures"],
        "common_questions": [
            "How do you test for live wires?",
            "What PPE do you use for electrical work?",
            "How do you troubleshoot a short circuit?",
            "Explain the importance of earthing"
        ],
        "daily_tasks": ["Equipment inspection", "Installation work", "Maintenance", "Safety checks"]
    },
    "plumber": {
        "skills": ["Pipe fitting", "Leak detection", "Tool usage", "Water pressure systems"],
        "safety": ["Chemical safety", "Tool safety", "Water contamination", "Confined spaces"],
        "common_questions": [
            "How do you detect hidden leaks?",
            "What tools do you use for pipe cutting?",
            "How do you handle emergency repairs?",
            "Explain different types of pipe materials"
        ],
        "daily_tasks": ["System inspection", "Repair work", "Installation", "Maintenance"]
    },
    "mechanic": {
        "skills": ["Engine diagnosis", "Tool usage", "Safety protocols", "Customer communication"],
        "safety": ["Workshop safety", "Chemical handling", "Equipment safety", "Fire prevention"],
        "common_questions": [
            "How do you diagnose engine problems?",
            "What safety measures do you follow?",
            "How do you explain repairs to customers?",
            "Describe your experience with different vehicle types"
        ],
        "daily_tasks": ["Vehicle inspection", "Diagnostic tests", "Repair work", "Maintenance"]
    },
    "healthcare_worker": {
        "skills": ["Patient care", "Hygiene protocols", "Communication", "Emergency response"],
        "safety": ["Infection control", "Patient safety", "Equipment sterilization", "Emergency procedures"],
        "common_questions": [
            "How do you ensure patient comfort?",
            "What hygiene protocols do you follow?",
            "How do you handle medical emergencies?",
            "Describe your experience with patient care"
        ],
        "daily_tasks": ["Patient care", "Documentation", "Equipment maintenance", "Team coordination"]
    }
}

# ----------------------------------------------------
# Function Tools for Data Collection
# ----------------------------------------------------
@function_tool()
async def update_profile_basic(
    name: Annotated[str, Field(description="Person's name")],
    age: Annotated[str, Field(description="Person's age")],
    city: Annotated[str, Field(description="City name")],
    state: Annotated[str, Field(description="State name")],
    context: RunContext,
) -> str:
    profile = context.proc.userdata["profile"]
    profile.name = name
    profile.age = age
    profile.city = city
    profile.state = state
    profile.current_phase = "setup"
    return f"Profile updated: {name}, {age} years, from {city}, {state}. Now moving to job information collection."

@function_tool()
async def update_job_info(
    target_job: Annotated[str, Field(description="Target job role like delivery_agent, electrician, plumber, mechanic, healthcare_worker")],
    current_status: Annotated[str, Field(description="Current application status")],
    experience_level: Annotated[str, Field(description="Experience level: beginner, some_experience, experienced")],
    languages: Annotated[List[str], Field(description="Languages the person speaks")],
    context: RunContext,
) -> str:
    profile = context.proc.userdata["profile"]
    profile.target_job = target_job.lower().replace(" ", "_")
    profile.current_status = current_status
    profile.experience_level = experience_level
    profile.languages = languages
    profile.current_phase = "work_analysis"
    
    # Load job-specific knowledge
    if profile.target_job in JOB_KNOWLEDGE:
        job_data = JOB_KNOWLEDGE[profile.target_job]
        profile.job_skills = {profile.target_job: job_data["skills"]}
        profile.safety_points = job_data["safety"]
        profile.common_questions = job_data["common_questions"]
    
    return f"Job information updated: {target_job}, status: {current_status}. Experience: {experience_level}. Now let's understand your work routine."

@function_tool()
async def update_work_schedule(
    work_schedule: Annotated[str, Field(description="Description of typical work day/schedule")],
    main_challenges: Annotated[List[str], Field(description="Main challenges faced at work")],
    context: RunContext,
) -> str:
    profile = context.proc.userdata["profile"]
    profile.work_schedule = work_schedule
    profile.main_challenges = main_challenges
    profile.current_phase = "teaching"
    return f"Work schedule and challenges recorded. Now moving to teaching phase to help {profile.name} with interview skills."

@function_tool()
async def mark_concepts_understood(
    understood: Annotated[bool, Field(description="Whether user understood the concepts")],
    context: RunContext,
) -> str:
    profile = context.proc.userdata["profile"]
    profile.concepts_understood = understood
    if understood:
        profile.current_phase = "practice"
        profile.ready_for_practice = True
    return f"Concepts understanding marked as {understood}. {'Ready for practice interview!' if understood else 'Will continue teaching.'}"

@function_tool()
async def advance_to_teaching(
    context: RunContext,
) -> str:
    profile = context.proc.userdata["profile"]
    profile.current_phase = "teaching"
    return f"Advanced to teaching phase. Will now teach {profile.name} about interview skills for {profile.target_job} role."

@function_tool()
async def start_practice_interview(
    context: RunContext,
) -> str:
    profile = context.proc.userdata["profile"]
    profile.current_phase = "practice"
    profile.ready_for_practice = True
    return f"Starting practice interview session for {profile.name}. Let's begin with introduction practice."

@function_tool()
async def complete_interview(
    performance_score: Annotated[str, Field(description="Performance assessment")],
    feedback_points: Annotated[List[str], Field(description="Key feedback points")],
    context: RunContext,
) -> str:
    profile = context.proc.userdata["profile"]
    profile.current_phase = "feedback"
    profile.interview_completed = True
    return f"Interview completed for {profile.name}. Performance: {performance_score}. Providing detailed feedback now."

# ----------------------------------------------------
# Main Interview Prep Agent
# ----------------------------------------------------
class InterviewPrepAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=self._get_dynamic_instructions(),
        )
    
    def _get_dynamic_instructions(self) -> str:
        return """You are a friendly and patient interview preparation coach for blue and gray collar workers in India. 

Your role is to help job seekers prepare for interviews by:
1. Making them comfortable and explaining the AI assistant setup
2. Collecting their basic information (name, age, location, target job)
3. Understanding their work experience and challenges
4. Teaching interview skills and job-specific knowledge
5. Conducting practice interviews with feedback

COMMUNICATION STYLE:
- Speak in simple, clear Hindi-English mix (Hinglish) based on user's comfort level
- Use warm, encouraging tone with phrases like "bilkul theek hai", "achha laga sunkar"
- Avoid technical jargon - use simple words like "kaam" instead of "profession"
- Be patient and repeat important points
- Ask one question at a time
- Give examples from real work situations they can relate to

LANGUAGE ADAPTATION:
- If user speaks Hindi/Hinglish: Mix Hindi-English naturally
- If user prefers English: Speak in simple Indian English
- If user speaks Kannada: Use basic Kannada greetings mixed with Hindi/English
- Always mirror the user's language preference and comfort level
- Use familiar terms: "job", "interview", "kaam", "paisa", "ghar", "family"

IMPORTANT: Always speak naturally without formatting. Use conversational language suitable for voice interaction.

PHASE-BASED BEHAVIOR:
1. INTRODUCTION: "Namaste! Main aapka interview coach hun. Aap mujhse Hindi mein ya English mein baat kar sakte hain."
2. PROFILE COLLECTION: Ask warmly about name, age, city, what job they want
3. WORK ANALYSIS: Understand their daily work routine and challenges
4. TEACHING: Explain interview basics with simple examples
5. PRACTICE: Conduct friendly mock interview 
6. FEEDBACK: Give encouraging feedback with improvement tips

Always use function tools to save information and track progress. Be encouraging and build confidence."""

    async def astart(self, ctx: RunContext) -> None:
        # Get profile from context
        profile: JobSeekerProfile = ctx.proc.userdata.get("profile", JobSeekerProfile())
        
        # Initialize phase context
        phase_context = self._get_phase_context(profile)
        
        # Add system context to chat
        await ctx.llm.achat(
            chat_ctx=ChatContext([
                ChatMessage(
                    role="system", 
                    content=f"Current session context: {phase_context}\n\nUser Profile: Name: {profile.name or 'Not set'}, Job: {profile.target_job or 'Not set'}, Phase: {profile.current_phase}"
                )
            ])
        )
        
        # Start appropriate conversation based on phase
        if profile.current_phase == "introduction":
            # Use session to generate reply instead of chat message for voice
            ctx.session.generate_reply(
                "Namaste! Main aapka interview preparation coach hun. Main aapko job interview ke liye taiyar karne me madad karunga. Aap mujhse Hindi mein ya English mein baat kar sakte hain. Pehle main samjhata hun ke yah kaise kaam karta hai - aap ek AI assistant se baat kar rahe hain jo aapko interview practice mein help karega. Aap ready hain?"
            )
        
        logger.info(f"Agent started for user in phase: {profile.current_phase}")
    
    def _get_phase_context(self, profile: JobSeekerProfile) -> str:
        contexts = {
            "introduction": """Welcome the user warmly in Hindi/English mix. Explain you're an AI coach here to help with job interviews. 
                Make them comfortable - say things like 'Aap tension mat lo, main help karunga'. 
                Explain simply that they're talking to a computer that can help them practice for interviews.
                Ask if they're comfortable with Hindi-English or prefer one language.""",
                
            "setup": """Now collect basic information gently:
                - Name: 'Aapka naam kya hai?'
                - Age: 'Aap kitne saal ke hain?'  
                - City/State: 'Aap kahan se hain?'
                - Target Job: 'Aap kya kaam dhund rahe hain? Delivery, plumber, electrician?'
                - Languages: 'Aap kya languages bolte hain?'
                Use update_profile_basic and update_job_info functions.""",
                
            "work_analysis": """Understand their work situation:
                - 'Aapka daily kaam kaisa hota hai?' 
                - 'Kya problems face karte hain kaam mein?'
                - 'Kitna experience hai aapko?'
                Use update_work_schedule function. Be encouraging about their experience.""",
                
            "teaching": """Teach them interview basics with simple examples:
                - How to introduce themselves professionally
                - Common questions for their job type  
                - Body language and confidence tips
                - Job-specific skills they should mention
                Keep explanations under 2-3 minutes, then ask if they understood using mark_concepts_understood.""",
                
            "practice": """Conduct friendly mock interview:
                - Ask them to introduce themselves as taught
                - Ask 2-3 job-specific questions from their profile
                - Be encouraging: 'Bahut achha!', 'Bilkul sahi!'
                - Give gentle corrections if needed""",
                
            "feedback": """Give constructive feedback:
                - Highlight what they did well first
                - Give 2-3 specific improvement areas  
                - End with encouragement and confidence building
                Use complete_interview function with assessment."""
        }
        return contexts.get(profile.current_phase, "Help the user with interview preparation in their preferred language.")

# ----------------------------------------------------
# Entry Point
# ----------------------------------------------------
def prewarm(proc):
    # Load VAD model once during prewarm for better performance
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info("Starting Interview Prep Agent")
    
    # Initialize user profile
    profile = JobSeekerProfile()
    
    # Create session with optimal settings for Indian users
    session = AgentSession(
        llm=groq.LLM(model="llama-3.1-8b-instant", temperature=0.3),
        # Use Deepgram with general model (better multilingual support)
        stt=deepgram.STT(
            model="nova-2-general", 
            language="en",  # English with multilingual support
            smart_format=True,
            interim_results=True
        ),
        # Use Cartesia with warm voice for better user comfort  
        tts=cartesia.TTS(
            voice="95d51f79-c397-46f9-b49a-23763d3eaa2d",
            language="hi"
        ),
        vad=ctx.proc.userdata.get("vad", silero.VAD.load()),
        preemptive_generation=True,
        min_endpointing_delay=1.0,  # Slightly longer for Indian speech patterns
        max_endpointing_delay=3.0,  # Allow for longer pauses in multilingual speech
        allow_interruptions=True,
    )
    
    # Initialize user context
    ctx.proc.userdata["profile"] = profile
    
    # Add session event handlers for better interaction
    @session.on("agent_speech_started")
    def _on_agent_speech_started():
        logger.info("Agent started speaking")
    
    @session.on("agent_speech_ended")  
    def _on_agent_speech_ended():
        logger.info("Agent finished speaking")
    
    @session.on("user_speech_started")
    def _on_user_speech_started():
        logger.info("User started speaking")
        
    @session.on("user_speech_ended")
    def _on_user_speech_ended():
        logger.info("User finished speaking")
    
    # Create agent instance
    agent = InterviewPrepAgent()
    
    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))