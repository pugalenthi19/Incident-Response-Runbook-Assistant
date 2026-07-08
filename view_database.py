from database import get_all_incidents

rows = get_all_incidents()

for row in rows:
    print(dict(row))