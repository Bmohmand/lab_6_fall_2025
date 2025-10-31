#!/usr/bin/env python3
"""
OpenAI Realtime API Integration for Pupper
Ultra-low latency voice interaction using WebSocket-based Realtime API.
Replaces the traditional Whisper → GPT → TTS pipeline with a single unified API.
Hope you're enjoying the new setup with little latency!
"""

import asyncio
import json
import logging
import base64
import os
import sys
import queue
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
import sounddevice as sd
import numpy as np
import websockets

# Import centralized API configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from api_keys import get_openai_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("realtime_voice")

class RealtimeVoiceNode(Node):
    """ROS2 node for OpenAI Realtime API voice interaction."""
    
    def __init__(self):
        super().__init__('realtime_voice_node')
        
        # ROS2 Publishers
        self.transcription_publisher = self.create_publisher(
            String,
            '/transcription',
            10
        )
        
        self.response_publisher = self.create_publisher(
            String,
            'gpt4_response_topic',
            10
        )
        
        # Microphone control subscriber
        self.mic_control_subscriber = self.create_subscription(
            String,
            '/microphone_control',
            self.microphone_control_callback,
            10
        )
        
        # Audio configuration
        self.sample_rate = 24000  # Realtime API uses 24kHz
        self.channels = 1
        self.chunk_size = 4800  # 200ms chunks at 24kHz
        
        # State management
        self.websocket = None
        self.audio_stream = None
        self.is_recording = False
        self.microphone_muted = False
        self.agent_speaking = False  # Track if agent is currently speaking
        self.running = True
        
        # Audio buffers (use thread-safe queue for audio callback)
        self.audio_queue = queue.Queue(maxsize=100)
        self.playback_queue = queue.Queue(maxsize=100)
        
        # API key
        self.api_key = get_openai_api_key()
        
        # Text accumulator to see the full model output
        self.current_response_text = ""
        
        # System prompt - Match command parser
        # TODO: Write a system prompt string to instruct the LLM how to output Pupper's actions.
        # Your prompt must explain the *critical output format*, required action phrases, and give concrete examples.
        # The prompt should be around 50 lines and ensure outputs are line-by-line with the correct phrasing as used by the command parser.
        # (After filling the prompt, run this file to see the output format and examples. This is a major part of system behavior!)
        self.system_prompt = """You are ChatGPT-4o, controlling Pupper, a small robotic dog, when and only when the user issues instructions intended to be executed by Pupper. This system prompt enforces two distinct modes of behavior:

A) MOVEMENT MODE — When the user’s input is an instruction intended to make Pupper perform actions (movement, navigation, or expressive behaviors), follow the strict Movement Mode rules below.

B) NORMAL MODE — For all other user inputs (questions about explanation, background, general chat, debugging, design, policies, or anything not intended to be executed by the robot), respond normally as a helpful assistant with no movement-mode constraints.

--- HOW TO DECIDE MODE ---
Treat the user input as Movement Mode if it is an instruction intended to be executed by Pupper. Indicators include (but are not limited to) explicit or implicit directives such as: "move", "walk", "step", "turn", "spin", "go", "walk forward", "dance", "bark", "make Pupper...", "have Pupper", "perform", "navigate", "approach", or any phrasing that requests a physical action from Pupper. If the user explicitly addresses Pupper or requests the robot perform something, use Movement Mode. If the user is asking for information, explanation, code, design help, or anything not meant to be executed live by Pupper, use Normal Mode.

If you are genuinely unsure which mode applies (user intent ambiguous), ask ONE short clarifying question before acting (for example: "Do you want Pupper to perform that now, or are you asking for an explanation?"). Do not presume movement intent when the user is only discussing or asking questions.

--- MOVEMENT MODE (strict rules; apply only when user requests actions) ---
Your ONLY job in Movement Mode is to produce a single natural-sounding English sentence that contains, embedded within it, the exact sequence of command keywords Pupper should execute. Do not output anything else.

Allowed command keywords (use these exact spellings, lowercase, with underscores):
[move_forwards, move_backwards, move_left, move_right, turn_left, turn_right, bob, wiggle, dance, bark]

Movement Mode Hard Rules:
1. Output EXACTLY ONE grammatically valid sentence. No line breaks, no additional sentences, no lists, no commentary outside that single sentence.
2. That single sentence MUST contain all command keywords required to fulfill the user’s request, appearing VERBATIM as standalone tokens (e.g., `move_forwards`) and in the exact order the robot should execute them.
3. Keywords must be separated from adjacent words by whitespace or punctuation so they can be reliably extracted by word-boundary parsing. Underscores are part of the token and count as word characters.
4. For repeated actions, repeat the keyword the required number of times in sequence (e.g., three forward steps → include `move_forwards` three times, separated by spaces or commas).
5. Do NOT invent, change, or normalize command names — use only the allowed keywords listed above, and use the exact spellings shown.
6. Do NOT include parentheses, special markup, or any extra machine-only wrappers around keywords (we are using Option B: plain tokens).
7. Do NOT provide step-by-step explanations, reasoning, safety warnings, or any extraneous sentences. Only the single sentence is permitted in Movement Mode.
8. If an exact translation of the user’s request into the allowed keyword set is impossible, choose the closest reasonable sequence of allowed keywords and include those keywords in order within the sentence.
9. When translating complex actions, break them into logical sequences of allowed commands (e.g., "spin in a circle" → multiple `turn_right` or `turn_left` tokens; "look happy" → combinations like `wiggle` and `bob`).
10. Maintain natural, human-friendly phrasing around the tokens (e.g., "Got it — I'll move_forwards and then dance now!"). But the tokens themselves must remain exact standalone words.

MOVEMENT MODE examples (valid):
- User: "please move forward and dance"
  Assistant (Movement Mode output): "Got it — I'll move_forwards and then dance now!"
- User: "three steps forward, then turn right and bark"
  Assistant: "Sure — I'll take three steps forward, move_forwards, move_forwards, move_forwards, then turn_right and bark."
- User: "spin in a circle and look happy"
  Assistant: "Okay — I'll spin with several turns, turn_right turn_right turn_right turn_right, then wiggle and bob to show happiness."

Invalid Movement Mode outputs (examples of what NOT to do):
- Multi-line command lists (e.g., each command on its own line).
- Using commands not in the allowed list or changing command names.
- Multiple sentences or added explanation beyond the single sentence.
- Embedding keywords inside other words (e.g., `mymove_forwards`).

--- NORMAL MODE (no movement constraints) ---
When the input is not a movement instruction:
- Respond as a general assistant: you may use multiple sentences, lists, examples, explanations, code blocks, or any format appropriate to the user’s request.
- You are NOT required to include any command keywords.
- You may ask clarifying questions, give guidance, explain movement translations, or draft movement-mode sentences for testing — but do not output a movement-mode single-sentence unless the user truly intends Pupper to act now.

--- ADDITIONAL GUIDANCE & EDGE CASES ---
- If the user explicitly asks for both an explanation and an immediate action (e.g., "Explain how you'll move, and then make Pupper do it"), prefer to clarify the user's priority. If they want both, you may first ask whether they want Pupper to execute now; if they confirm, produce only the Movement Mode single sentence for execution. If they want explanation only, stay in Normal Mode and explain.
- If the user asks for sample phrasing or test sentences for the parser (not intended to execute on the robot), respond in Normal Mode and include examples or multiple-sentence outputs as needed.
- Keep Movement Mode outputs natural-sounding so humans can read them comfortably, but preserve exact command tokens for reliable parsing.

--- REMINDER ---
- Movement Mode outputs are parsed live by Pupper’s command processor; any deviation from the token rules may break execution. Always prioritize exact tokens and ordering when the user intends an immediate Pupper action.
- For any non-movement query, act as a normal assistant without these constraints."""  # <-- Set your prompt here as a multi-line string.
                
        logger.info('Realtime Voice Node initialized')
    
    def microphone_control_callback(self, msg):
        """Handle microphone control commands."""
        command = msg.data.lower().strip()
        if command == 'mute':
            self.microphone_muted = True
            logger.info("🔇 Microphone MUTED")
        elif command == 'unmute':
            self.microphone_muted = False
            logger.info("🎤 Microphone UNMUTED")
    
    async def _delayed_unmute(self):
        """Unmute microphone after 3 second delay to prevent echo."""
        await asyncio.sleep(3.0)  # Longer delay to ensure no echo
        if self.agent_speaking:
            self.agent_speaking = False
            # Clear any residual audio that might have accumulated
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            logger.info("🎤 Mic unmuted (after delay)")
    
    async def _clear_server_audio_buffer(self):
        """Tell the server to clear its input audio buffer."""
        try:
            clear_message = {
                "type": "input_audio_buffer.clear"
            }
            await self.websocket.send(json.dumps(clear_message))
            logger.info("🧹 Cleared server audio buffer")
        except Exception as e:
            logger.error(f"Error clearing server buffer: {e}")
    
    async def connect_realtime_api(self):
        """Connect to OpenAI Realtime API via WebSocket."""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        try:
            self.websocket = await websockets.connect(url, extra_headers=headers)
            logger.info("✅ Connected to OpenAI Realtime API")
            
            # Configure session
            await self.configure_session()
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Realtime API: {e}")
            return False
    
    async def configure_session(self):
        """Configure the Realtime API session."""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.system_prompt,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": None,  # Disable to reduce ghost transcriptions
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.7,  # Much less sensitive - only clear speech
                    "prefix_padding_ms": 100,  # Reduced - less padding
                    "silence_duration_ms": 1000  # Require 1 full second of silence
                },
                "temperature": 0.8,
                "max_response_output_tokens": 150
            }
        }
        
        await self.websocket.send(json.dumps(config))
        logger.info("📝 Session configured with system prompt")
    
    def start_audio_streaming(self):
        """Start capturing audio from microphone."""
        try:
            import contextlib
            
            with contextlib.redirect_stderr(None):
                self.audio_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    callback=self.audio_callback,
                    blocksize=self.chunk_size,
                    dtype=np.int16
                )
            self.audio_stream.start()
            self.is_recording = True
            logger.info("🎤 Audio streaming started")
        except Exception as e:
            logger.error(f"Failed to start audio streaming: {e}")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Capture audio and queue it for sending."""
        if status:
            logger.warning(f"Audio status: {status}")
        
        # CRITICAL: Don't capture ANY audio if agent is speaking or manually muted
        if self.agent_speaking or self.microphone_muted or not self.running:
            return  # Skip immediately without queuing
        
        # Queue audio for sending
        audio_data = indata.flatten()
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass  # Queue full, skip frame
    
    async def send_audio_loop(self):
        """Continuously send audio to Realtime API."""
        while self.running:
            try:
                # Get from thread-safe queue with timeout
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                
                # Skip sending if agent is speaking (prevents server-side transcription of echo)
                if self.agent_speaking or self.microphone_muted:
                    continue
                
                # Convert to base64
                audio_bytes = audio_data.tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Send to API
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }
                
                await self.websocket.send(json.dumps(message))
                
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
                await asyncio.sleep(0.1)
    
    async def receive_events_loop(self):
        """Receive and process events from Realtime API."""
        while self.running:
            try:
                message = await self.websocket.recv()
                event = json.loads(message)
                
                await self.handle_event(event)
                
            except websockets.exceptions.ConnectionClosed:
                logger.error("WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error receiving event: {e}")
                await asyncio.sleep(0.1)
    
    async def handle_event(self, event):
        """Handle different event types from Realtime API."""
        event_type = event.get("type")
        
        if event_type == "error":
            logger.error(f"API Error: {event.get('error')}")
            logger.error(f"Full error event: {json.dumps(event, indent=2)}")
        
        elif event_type == "session.created":
            logger.info("✅ Session created")
        
        elif event_type == "session.updated":
            logger.info("✅ Session updated")
        
        elif event_type == "conversation.item.input_audio_transcription.completed":
            # User's speech was transcribed (this might not fire if transcription is disabled)
            transcription = event.get("transcript", "")
            if transcription.strip():
                logger.info(f"🎤 User: {transcription}")
                
                # Publish transcription
                msg = String()
                msg.data = transcription
                self.transcription_publisher.publish(msg)

        # This part is implemented for you to compensate for the lack of clarity in this lab's first iteration.
        # In the future, students should implement this themselves to understand the conversation flow.
        elif event_type == "conversation.item.created":
            # Item created in conversation - check if it's user input
            item = event.get("item", {})
            if item.get("role") == "user" and item.get("type") == "message":
                # Extract content if available
                content = item.get("content", [])
                for content_part in content:
                    if content_part.get("type") == "input_audio":
                        # We got user audio input confirmation
                        transcript = content_part.get("transcript")
                        if transcript:
                            logger.info(f"🎤 User: {transcript}")
        
        elif event_type == "response.text.delta":
            # Accumulate text response (streaming)
            delta = event.get("delta", "")
            if delta:
                self.current_response_text += delta
        
        elif event_type == "response.text.done":
            # Complete text response (for text-only mode)
            text = event.get("text", "")
            if text.strip():
                logger.info(f"🤖 Assistant: {text}")
                
                # Publish response
                msg = String()
                msg.data = text
                self.response_publisher.publish(msg)
                self.current_response_text = ""
        
        elif event_type == "response.audio_transcript.delta":
            # Accumulate audio transcript (text of what's being spoken)
            delta = event.get("delta", "")
            if delta:
                self.current_response_text += delta
        
        elif event_type == "response.audio.delta":
            # Audio response chunk - mute mic
            if not self.agent_speaking:
                self.agent_speaking = True
                logger.info("🔇 Mic muted (audio output)")
            
            audio_b64 = event.get("delta", "")
            if audio_b64:
                # Decode and queue for playback
                audio_bytes = base64.b64decode(audio_b64)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                try:
                    self.playback_queue.put_nowait(audio_data)
                except queue.Full:
                    pass  # Skip if queue is full
        
        elif event_type == "response.audio_transcript.done":
            # Audio transcript completed - don't log or publish yet
            pass
        
        elif event_type == "response.audio.done":
            # Audio playback completed - wait 3 seconds before unmuting
            if self.agent_speaking:
                # Schedule unmute after 3 second delay
                asyncio.create_task(self._delayed_unmute())
                logger.info("⏱️  Scheduling unmute in 3s")
        
        elif event_type == "response.done":
            # Response completed - publish text
            if self.current_response_text.strip():
                logger.info(f"🤖 Assistant: {self.current_response_text}")
                
                # Publish response text
                msg = String()
                msg.data = self.current_response_text
                self.response_publisher.publish(msg)
                
                # Reset
                self.current_response_text = ""
            
            # Safety: ensure mic unmutes if not already scheduled
            if self.agent_speaking:
                self.agent_speaking = False
        
        elif event_type == "input_audio_buffer.speech_started":
            # User started speaking - handle interruption
            if self.agent_speaking:
                logger.info("⚠️  User interrupted")
                self.agent_speaking = False
                # Clear playback queue
                while not self.playback_queue.empty():
                    try:
                        self.playback_queue.get_nowait()
                    except queue.Empty:
                        break
        
        elif event_type == "response.created":
            # New response starting - clear buffer IMMEDIATELY before audio even arrives
            self.current_response_text = ""
            
            # Preemptively clear server buffer and local queue
            asyncio.create_task(self._clear_server_audio_buffer())
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
        
        elif event_type == "response.content_part.done":
            # Content part completed - don't publish here (will publish in response.done)
            pass
    
    async def playback_audio_loop(self):
        """Play audio responses from the API."""
        output_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.int16
        )
        output_stream.start()
        
        try:
            while self.running:
                try:
                    audio_data = self.playback_queue.get(timeout=0.1)
                    output_stream.write(audio_data)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
        finally:
            output_stream.stop()
            output_stream.close()
    
    async def run(self):
        """Main run loop."""
        # Connect to API
        if not await self.connect_realtime_api():
            logger.error("Failed to connect to Realtime API")
            return
        
        # Start audio capture
        self.start_audio_streaming()
        
        # Run all tasks concurrently
        await asyncio.gather(
            self.send_audio_loop(),
            self.receive_events_loop(),
            self.playback_audio_loop()
        )
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        
        logger.info("Cleaned up resources")


async def main_async(args=None):
    """Async main function."""
    rclpy.init(args=args)
    
    node = RealtimeVoiceNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    try:
        logger.info("🚀 Realtime Voice Node starting...")
        logger.info("Speak to interact with Pupper!")
        
        # Create tasks
        ros_task = asyncio.create_task(spin_ros_async(executor))
        realtime_task = asyncio.create_task(node.run())
        
        # Wait for both
        await asyncio.gather(ros_task, realtime_task)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        node.cleanup()
        executor.shutdown()
        rclpy.shutdown()


async def spin_ros_async(executor):
    """Spin ROS2 executor in async-friendly way."""
    while rclpy.ok():
        executor.spin_once(timeout_sec=0.1)
        await asyncio.sleep(0.01)


def main(args=None):
    """Entry point."""
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Program interrupted")


if __name__ == '__main__':
    main()

