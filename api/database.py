import os
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./prediction_logs.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    label = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    fake_probability = Column(Float, nullable=False)
    real_probability = Column(Float, nullable=False)
    model_name = Column(String(128), nullable=False)
    source_url = Column(Text, nullable=True)
    text_preview = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def log_prediction(text: str, prediction: dict, source_url: str | None = None) -> None:
    with session_scope() as session:
        session.add(
            PredictionLog(
                label=prediction["label"],
                confidence=prediction["confidence"],
                fake_probability=prediction["fake_probability"],
                real_probability=prediction["real_probability"],
                model_name=prediction["model_name"],
                source_url=source_url,
                text_preview=text[:500],
                word_count=prediction["stats"]["words"],
            )
        )


def recent_predictions(limit: int = 25) -> list[dict]:
    with session_scope() as session:
        rows = (
            session.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .limit(min(limit, 100))
            .all()
        )
        return [
            {
                "id": row.id,
                "created_at": row.created_at.isoformat(),
                "label": row.label,
                "confidence": row.confidence,
                "fake_probability": row.fake_probability,
                "real_probability": row.real_probability,
                "model_name": row.model_name,
                "source_url": row.source_url,
                "text_preview": row.text_preview,
                "word_count": row.word_count,
            }
            for row in rows
        ]
