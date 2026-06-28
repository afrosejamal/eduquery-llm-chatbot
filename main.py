from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

import json
import requests
import sqlite3
import uuid
import tempfile
import os

from backend.document_processor import process_document
from backend.session_store import temp_collections
from backend.rag_engine import (
    answer_query,
    generate_test,
    generate_flashcards,
    generate_interview_questions
)
from backend.database import create_table, get_connection

print("🔥 main.py loaded")

app = FastAPI(title="EduQuery Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_table()

# ============================
# REQUEST & RESPONSE MODELS
# ============================

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  

class QueryResponse(BaseModel):
    answer: str
    refined: str
    images: List[str]
    sources: List[str]
    excerpts: Optional[List[dict]] = []
    suggestions: Optional[List[str]] = []


class TestRequest(BaseModel):
    topic: str


class AnswerItem(BaseModel):
    question: str
    correct: str
    user: str


class EvalRequest(BaseModel):
    answers: List[AnswerItem]


class SaveTestRequest(BaseModel):
    topic: str
    score: int
    total: int
    answers: List[dict]


# ============================
# ROOT
# ============================

@app.get("/")
def root():
    return {"status": "EduQuery backend running"}


# ============================
# SMART CHATBOT ENDPOINT
# ============================

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):

    try:
        result = answer_query(request.question, session_id=request.session_id)

        return {
            "answer": result["answer"],
            "refined": result["refined"],
            "images": result["images"],
            "sources": result["sources"],
            "excerpts": result.get("excerpts", []),
            "suggestions": result.get("suggestions", [])
        }
    
    except Exception as e:
        print("❌ ERROR in /ask:", e)
        return {
            "answer": "Internal server error",
            "refined": "",
            "images": [],
            "sources": []
        }


# ============================
# TEST GENERATION
# ============================

print("🔥 registering generate_test endpoint")

@app.post("/generate_test")
def create_test(req: TestRequest):
    questions = generate_test(req.topic)
    return {"questions": questions}


# ============================
# HISTORY
# ============================

@app.get("/history")
def get_history():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT question, answer, sources, timestamp FROM query_history ORDER BY id DESC"
    )

    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "question": row[0],
            "answer": row[1],
            "sources": json.loads(row[2]) if row[2] else [],
            "timestamp": row[3]
        })

    return history


# ============================
# TEST EVALUATION
# ============================

@app.post("/evaluate_test")
def evaluate_test(req: EvalRequest):

    score = 0
    results = []

    for item in req.answers:
        user_answer = item.user.strip().lower()
        correct_answer = item.correct.strip().lower()

        if not user_answer:
            results.append({
                "question": item.question,
                "correct": False,
                "reason": "Not answered"
            })
            continue

        prompt = f"""
Question: {item.question}
Correct answer: {item.correct}
Student answer: {item.user}

Is the student's answer correct? Reply only YES or NO.
"""

        try:
            from backend.rag_engine import _groq_client, GROQ_MODEL
            eval_response = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=5
            )
            result = eval_response.choices[0].message.content.strip().upper()

            if "YES" in result:
                score += 1
                results.append({"question": item.question, "correct": True})
            else:
                results.append({"question": item.question, "correct": False})

        except:
            if user_answer in correct_answer:
                score += 1
                results.append({"question": item.question, "correct": True})
            else:
                results.append({"question": item.question, "correct": False})

    return {
        "score": score,
        "total": len(req.answers),
        "results": results
    }


# ============================
# SAVE TEST RESULT
# ============================

@app.post("/save_test_result")
def save_test_result(req: SaveTestRequest):

    conn = get_connection()
    cursor = conn.cursor()

    percentage = (req.score / req.total) * 100 if req.total else 0

    cursor.execute(
        "INSERT INTO test_results (topic, score, total_questions, percentage, answers_data) VALUES (?, ?, ?, ?, ?)",
        (req.topic, req.score, req.total, percentage, json.dumps(req.answers))
    )

    conn.commit()
    conn.close()

    return {"status": "saved"}


# ============================
# analytics
# ============================
@app.get("/analytics")
def get_analytics():
    try:
        conn = sqlite3.connect("eduquery.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all test results
        cursor.execute("""
            SELECT topic, score, total_questions, percentage, timestamp
            FROM test_results
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()

        if not rows:
            return {
                "total_tests": 0,
                "average_score": 0,
                "average_percentage": 0,
                "best_score": "0/0",
                "best_percentage": 0,
                "best_topic": "",
                "weak_topics": [],
                "strong_topics": [],
                "recent_tests": [],
                "progress_over_time": []
            }

        total_tests = len(rows)
        total_score = sum(row["score"] for row in rows)
        total_percentage = sum(row["percentage"] for row in rows)
        avg_score = round(total_score / total_tests, 1)
        avg_percentage = round(total_percentage / total_tests, 1)

        # Best test
        best_row = max(rows, key=lambda r: r["percentage"])
        best_score_display = f"{best_row['score']}/{best_row['total_questions']}"
        best_percentage = round(best_row["percentage"], 1)
        best_topic = best_row["topic"]

        # Topic stats
        topic_map = {}
        for row in rows:
            t = row["topic"]
            if t not in topic_map:
                topic_map[t] = {"total_score": 0, "total_questions": 0, "count": 0}
            topic_map[t]["total_score"] += row["score"]
            topic_map[t]["total_questions"] += row["total_questions"]
            topic_map[t]["count"] += 1

        topic_averages = []
        for topic, stats in topic_map.items():
            avg_pct = (stats["total_score"] / stats["total_questions"]) * 100 if stats["total_questions"] else 0
            topic_averages.append({"topic": topic, "average": round(avg_pct, 1)})

        topic_averages.sort(key=lambda x: x["average"], reverse=True)

        strong_topics = [t for t in topic_averages if t["average"] >= 70][:3]
        weak_topics = [t for t in topic_averages if t["average"] < 70]
        weak_topics.sort(key=lambda x: x["average"])
        weak_topics = weak_topics[:3]

        # Recent tests (last 5)
        recent_tests = []
        for row in rows[:5]:
            recent_tests.append({
                "topic": row["topic"],
                "score": f"{row['score']}/{row['total_questions']}",
                "percentage": round(row["percentage"], 1),
                "timestamp": row["timestamp"]
            })

        # Progress over time (last 7 tests, oldest first)
        progress = []
        for row in reversed(rows[:7]):
            progress.append({
                "date": row["timestamp"][:10] if row["timestamp"] else "Unknown",
                "percentage": round(row["percentage"], 1),
                "topic": row["topic"][:15] + "..." if len(row["topic"]) > 15 else row["topic"]
            })

        conn.close()

        return {
            "total_tests": total_tests,
            "average_score": avg_score,
            "average_percentage": avg_percentage,
            "best_score": best_score_display,
            "best_percentage": best_percentage,
            "best_topic": best_topic,
            "weak_topics": weak_topics,
            "strong_topics": strong_topics,
            "recent_tests": recent_tests,
            "progress_over_time": progress
        }

    except Exception as e:
        print("❌ Analytics error:", e)
        return {"error": str(e)}
        


# ============================
# upload document
# ============================


@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...), session_id: str = None):
    try:
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        collection_name = f"temp_{session_id}"
        num_chunks = process_document(tmp_path, collection_name)
        
        temp_collections[session_id] = {
            "collection_name": collection_name,
            "filename": file.filename
        }
        
        os.unlink(tmp_path)
        
        return {
            "session_id": session_id,
            "message": f"Document '{file.filename}' processed into {num_chunks} chunks",
            "status": "ready"
        }
    except Exception as e:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    

# ============================
# FLASHCARD GENERATION
# ============================
class FlashcardRequest(BaseModel):
    topic: str
    num_cards: int = 5

@app.post("/generate_flashcards")
def generate_flashcards_endpoint(req: FlashcardRequest):
    flashcards = generate_flashcards(req.topic, req.num_cards)
    if not flashcards:
        return {"error": "Could not generate flashcards"}
    # Save to database
    conn = get_connection()
    cursor = conn.cursor()
    for card in flashcards:
        cursor.execute(
            "INSERT INTO flashcards (topic, question, answer) VALUES (?, ?, ?)",
            (req.topic, card['question'], card['answer'])
        )
    conn.commit()
    conn.close()
    return {"flashcards": flashcards}


# ============================
# Get due flashcards for study
# ============================
@app.get("/flashcards/due")
def get_due_flashcards(limit: int = 20):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, topic, question, answer FROM flashcards WHERE next_review <= datetime('now') ORDER BY random() LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return {"flashcards": [{"id": r[0], "topic": r[1], "question": r[2], "answer": r[3]} for r in rows]}

# ============================
# Submit review result
# ============================
class ReviewRequest(BaseModel):
    card_id: int
    quality: int  # 0-5 (0 = fail, 5 = perfect)

@app.post("/flashcards/review")
def review_flashcard(req: ReviewRequest):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT ease_factor, interval, repetitions FROM flashcards WHERE id = ?", (req.card_id,))
    row = cursor.fetchone()
    if not row:
        return {"error": "Card not found"}
    ef, interval, reps = row

    # SM‑2 algorithm (simplified)
    if req.quality >= 3:
        if reps == 0:
            interval = 1
        elif reps == 1:
            interval = 6
        else:
            interval = round(interval * ef)
        reps += 1
    else:
        reps = 0
        interval = 1
        ef = max(1.3, ef - 0.2)

    # Update card
    cursor.execute(
        "UPDATE flashcards SET ease_factor = ?, interval = ?, repetitions = ?, next_review = datetime('now', '+' || ? || ' days'), last_reviewed = datetime('now') WHERE id = ?",
        (ef, interval, reps, interval, req.card_id)
    )
    conn.commit()
    conn.close()
    return {"status": "updated"}


class InterviewRequest(BaseModel):
    topic: str
    num_questions: int = 3
    difficulty: str = "medium"

@app.post("/generate_interview_questions")
def generate_interview_questions_endpoint(req: InterviewRequest):
    questions = generate_interview_questions(req.topic, req.num_questions, req.difficulty)
    if not questions:
        return {"error": "Could not generate questions"}
    return {"questions": questions}
