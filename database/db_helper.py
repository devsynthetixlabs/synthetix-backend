from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import UserDefinedType
from sqlalchemy import event

# 1. Define the Vector type for SQLAlchemy
class PGVector(UserDefinedType):
    def get_col_spec(self, **kw):
        return "VECTOR"

# 2. Register it so the inspector doesn't crash
@event.listens_for(postgresql.base.PGDialect, "reflection_compiler_setup")
def register_vector_type(dialect, compiler):
    # This maps the 'vector' type in Postgres to our dummy PGVector class
    dialect.ischema_names["vector"] = PGVector

def get_db_schema(engine):
    inspector = inspect(engine)
    schema_info = []
    
    # Iterate through every table in the database
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        col_names = [f"{col['name']} ({col['type']})" for col in columns]
        schema_info.append(f"Table: {table_name}\nColumns: {', '.join(col_names)}")
    
    return "\n\n".join(schema_info)