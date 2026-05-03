import json
import boto3
from django.http import StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from .rag import retrieve_results, parse_results, generate_with_ollama, generate_with_bedrock, is_ollama_running, knowledgeBaseId


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

            # Route to Ollama or Bedrock based on availability
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
    polly_client = boto3.client('polly', region_name='eu-north-1')

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


def about(request):
    return render(request, 'about.html')


def tech_stack(request):
    return render(request, 'tech_stack.html')