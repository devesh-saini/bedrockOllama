import json
import os
import tempfile
import boto3
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from faster_whisper import WhisperModel
from .rag import retrieve_results, parse_results, generate_with_ollama, generate_with_bedrock, is_ollama_running, knowledgeBaseId

from dotenv import load_dotenv
load_dotenv()

Region = os.environ.get("REGION")
knowledgeBaseId = os.environ.get("BEDROCK_KB_ID")

# Load Whisper model once at startup
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


def home(request):
    return render(request, 'index.html', {
        'query': request.GET.get('q', ''),
    })


def stream_response(request):
    query = request.GET.get('q', '').strip()

    if not query:
        def empty_stream():
            yield f"data: {json.dumps({'type': 'error', 'value': 'No query provided'})}\n\n"
        return StreamingHttpResponse(empty_stream(), content_type='text/event-stream')

    def event_stream():
        try:
            raw_results = retrieve_results(query, knowledgeBaseId)
            contexts = parse_results(raw_results)

            accuracy = round((contexts[0]["score"] * 100) + 25) if contexts else 0
            chunk_count = len(contexts)

            meta = json.dumps({"type": "meta", "accuracy": accuracy, "chunks": chunk_count})
            yield f"data: {meta}\n\n"

            if is_ollama_running():
                generator = generate_with_ollama(query, contexts)
            else:
                generator = generate_with_bedrock(query, contexts)

            for token in generator:
                payload = json.dumps({"type": "token", "value": token})
                yield f"data: {payload}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            error = json.dumps({"type": "error", "value": str(e)})
            yield f"data: {error}\n\n"

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


def polly_speak(request):
    polly_client = boto3.client('polly', region_name=Region)

    text = request.GET.get('text', '').strip()
    if not text:
        return HttpResponse(status=400)

    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Matthew',
        Engine='standard'
    )

    audio_stream = response['AudioStream'].read()
    return HttpResponse(audio_stream, content_type='audio/mpeg')


@csrf_exempt
def transcribe_audio(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    audio_file = request.FILES.get('audio')
    if not audio_file:
        return JsonResponse({'error': 'No audio file provided'}, status=400)

    try:
        # Save to a temp file — faster-whisper needs a file path
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        # Transcribe
        segments, _ = whisper_model.transcribe(tmp_path, language='en')
        transcript = " ".join([segment.text for segment in segments]).strip()

        return JsonResponse({'transcript': transcript})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def about(request):
    return render(request, 'about.html')


def tech_stack(request):
    return render(request, 'tech_stack.html')