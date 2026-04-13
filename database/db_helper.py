from sqlalchemy import inspect

def get_db_schema(engine):
    inspector = inspect(engine)
    schema_info = []
    
    # Iterate through every table in the database
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        col_names = [f"{col['name']} ({col['type']})" for col in columns]
        schema_info.append(f"Table: {table_name}\nColumns: {', '.join(col_names)}")
    
    return "\n\n".join(schema_info)