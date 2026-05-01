from sqlalchemy import inspect

def get_db_schema(engine):
    inspector = inspect(engine)
    schema_info = []
    
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        col_info = []
        for col in columns:
            col_name = col['name']
            col_type = str(col['type'])
            col_info.append(f"{col_name} ({col_type})")
        
        schema_info.append(f"Table: {table_name}\nColumns: {', '.join(col_info)}")
        
        foreign_keys = inspector.get_foreign_keys(table_name)
        if foreign_keys:
            fk_info = [f"{fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}" for fk in foreign_keys]
            schema_info.append(f"Foreign Keys: {', '.join(fk_info)}")
    
    return "\n\n".join(schema_info)