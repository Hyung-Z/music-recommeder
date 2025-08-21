import json
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import os

# --- 설정 ---
VERSION = "electra" # 또는 v1.3 등 사용하시는 버전에 맞게 수정
TOP_K = 3
DEFAULT_EMOTION_WEIGHT = 0.4
DEFAULT_CONTEXT_WEIGHT = 0.6

# --- 파일 경로 정의 ---
EMOTION_MODEL_PATH = f'./my_emotion_model_{VERSION}'
MUSIC_DB_SONG_INFO_PATH = f'./lyrics_recommend/song_database_{VERSION}.json'
MUSIC_DB_EMOTION_VECTORS_PATH = f'./lyrics_recommend/lyrics_emotion_embeddings_{VERSION}.npy'
MUSIC_DB_CONTEXT_VECTORS_PATH = f'./lyrics_recommend/lyrics_context_embeddings_KoSimCSE.npy'

FLOWER_DB_FLOWER_INFO_PATH = f'./flower_recommend/flower_database_electra.json'
FLOWER_DB_EMOTION_VECTORS_PATH = f'./flower_recommend/flower_emotion_embeddings_electra.npy'
FLOWER_DB_CONTEXT_VECTORS_PATH = f'./flower_recommend/flower_context_embeddings_KoSimCSE.npy'


# --- 모델 및 데이터 로드 ---
print("서버 시작: 모델과 데이터베이스를 로드합니다...")
try:
    emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH, output_hidden_states=True)
    emotion_model.eval()

    ## BM-K/KoSimCSE-roberta or jhgan/ko-sbert-nli
    context_model = SentenceTransformer('BM-K/KoSimCSE-roberta')
    music_emotion_vectors = np.load(MUSIC_DB_EMOTION_VECTORS_PATH)
    music_context_vectors = np.load(MUSIC_DB_CONTEXT_VECTORS_PATH)
    with open(MUSIC_DB_SONG_INFO_PATH, 'r', encoding='utf-8') as f:
        music_db_info = json.load(f)
    print("- 음악 DB 로드 완료!")
    
    flower_emotion_vectors = np.load(FLOWER_DB_EMOTION_VECTORS_PATH)
    flower_context_vectors = np.load(FLOWER_DB_CONTEXT_VECTORS_PATH)
    with open(FLOWER_DB_FLOWER_INFO_PATH, 'r', encoding='utf-8') as f:
        flower_db_info = json.load(f)
    print("- 꽃 DB 로드 완료!")

except FileNotFoundError as e:
    print(f"오류: 필수 파일을 찾을 수 없습니다. ({e})")
    print("모든 파일이 app.py와 같은 폴더 구조에 있는지 확인하세요.")
    exit()

# --- Flask 앱 생성 ---
app = Flask(__name__)

def cosine_similarity(vec1, vec2_matrix):
    vec1_norm = np.linalg.norm(vec1)
    vec2_matrix_norm = np.linalg.norm(vec2_matrix, axis=1)
    epsilon = 1e-8
    return np.dot(vec2_matrix, vec1) / (vec1_norm * vec2_matrix_norm + epsilon)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/flower')
def flower_page():
    return render_template('flower.html')


@app.route('/recommend_music', methods=['POST'])
def recommend_music_api():
    data = request.json
    user_input = data.get('text')
    emotion_weight = data.get('emotion_weight', 0.4)
    context_weight = data.get('context_weight', 0.6)

    if not user_input:
        return jsonify({"error": "텍스트를 입력해주세요."}), 400

    inputs = emotion_tokenizer(user_input, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    
    user_emotion_label = emotion_model.config.id2label[torch.argmax(outputs.logits, dim=-1).item()]
    user_emotion_vector = outputs.hidden_states[-1][:, 0, :].numpy().flatten()
    user_context_vector = context_model.encode(user_input)

    candidate_indices = [i for i, song in enumerate(music_db_info) if song['emotion'] == user_emotion_label]
    if not candidate_indices:
        return jsonify({"emotion": user_emotion_label, "recommendations": []})

    cand_emotion_vectors = music_emotion_vectors[candidate_indices]
    cand_context_vectors = music_context_vectors[candidate_indices]
    
    emotion_sims = cosine_similarity(user_emotion_vector, cand_emotion_vectors)
    context_sims = cosine_similarity(user_context_vector, cand_context_vectors)
    total_scores = (emotion_weight * emotion_sims) + (context_weight * context_sims)
    
    sorted_indices = np.argsort(total_scores)[::-1]
    
    recommendations = []
    for i in range(min(TOP_K, len(sorted_indices))):
        original_idx = candidate_indices[sorted_indices[i]]
        song = music_db_info[original_idx]
        score = total_scores[sorted_indices[i]]
        recommendations.append({
            "rank": i + 1, "artist": song['artist'], "title": song['title'],
            "score": float(score), "videoId": song.get('videoId', '')
        })
    return jsonify({"emotion": user_emotion_label, "recommendations": recommendations})


@app.route('/recommend_flower', methods=['POST'])
def recommend_flower_api():
    data = request.json
    user_input = data.get('text')
    emotion_weight = data.get('emotion_weight', 0.2)
    context_weight = data.get('context_weight', 0.8)

    if not user_input:
        return jsonify({"error": "텍스트를 입력해주세요."}), 400

    inputs = emotion_tokenizer(user_input, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    
    user_emotion_label = emotion_model.config.id2label[torch.argmax(outputs.logits, dim=-1).item()]
    user_emotion_vector = outputs.hidden_states[-1][:, 0, :].numpy().flatten()
    user_context_vector = context_model.encode(user_input)

    candidate_indices = [i for i, flower in enumerate(flower_db_info) if flower['emotion'] == user_emotion_label]
    if not candidate_indices:
        return jsonify({"emotion": user_emotion_label, "recommendations": []})

    cand_emotion_vectors = flower_emotion_vectors[candidate_indices]
    cand_context_vectors = flower_context_vectors[candidate_indices]
    
    emotion_sims = cosine_similarity(user_emotion_vector, cand_emotion_vectors)
    context_sims = cosine_similarity(user_context_vector, cand_context_vectors)
    total_scores = (emotion_weight * emotion_sims) + (context_weight * context_sims)
    
    sorted_indices = np.argsort(total_scores)[::-1]
    
    recommendations = []
    for i in range(min(TOP_K, len(sorted_indices))):
        original_idx = candidate_indices[sorted_indices[i]]
        flower = flower_db_info[original_idx]
        score = total_scores[sorted_indices[i]]
        recommendations.append({
            "rank": i + 1, "name": flower['name'], "meaning": flower['words'],
            "score": float(score)
        })
    return jsonify({"emotion": user_emotion_label, "recommendations": recommendations})

# Hugging Face Spaces는 기본적으로 7860 포트를 사용합니다.
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
