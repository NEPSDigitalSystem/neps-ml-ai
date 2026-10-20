from sqlalchemy import create_engine
from app.config import get_settings

settings = get_settings()

# Create SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    echo=settings.APP_ENV == "development",
)
