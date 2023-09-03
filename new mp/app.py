from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from rouge_score import rouge_scorer

app = Flask(__name__)

def calculate_rouge(reference_summary, generated_summary):
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = rouge.score(reference_summary, generated_summary)
    return scores

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    video_url = data.get('videoUrl')
    video_id = video_url.split("=")[1]

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_transcript = " ".join([segment["text"] for segment in transcript])
    summarization_model = "sshleifer/distilbart-cnn-12-6"
    summarization_revision = "a4f8f3e"
    summarizer = pipeline('summarization', model=summarization_model, revision=summarization_revision)
    
    segment_size = 1000
    num_iters = (len(full_transcript) - 1) // segment_size + 1
    summarized_text = []

    for i in range(num_iters):
        start = i * segment_size
        end = (i + 1) * segment_size
        segment = full_transcript[start:end]

        summary = summarizer(segment)
        summary_text = summary[0]['summary_text']

        summarized_text.append(summary_text)

    reference_summary = full_transcript
    generated_summary = ' '.join(summarized_text)

    rouge_scores = calculate_rouge(reference_summary, generated_summary)
    
    # Prepare the results
    results = {
        'fullTranscript': full_transcript,
        'summarizedText': generated_summary,
        'originalTranscriptLength': len(reference_summary),
        'summarizedTextLength': len(generated_summary),
        'rougeScore': (rouge_scores['rougeL'].precision)*100
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
