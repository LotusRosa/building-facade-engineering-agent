import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.path = self.root/'audit.sqlite3'
        with self.transaction() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS entities(kind TEXT, id TEXT PRIMARY KEY, project TEXT, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS entities_kind ON entities(kind,project);
                CREATE TABLE IF NOT EXISTS submissions(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, response TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT NOT NULL);
            ''')

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path,timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def get(db, kind, id):
        row = db.execute('SELECT data FROM entities WHERE kind=? AND id=?',(kind,id)).fetchone()
        if row is None:
            raise ValueError('Unknown ' + kind + ': ' + str(id))
        return json.loads(row[0])

    @staticmethod
    def put(db,kind,data):
        db.execute('INSERT INTO entities(kind,id,project,data) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data',
                   (kind,data['id'],data.get('project',data['id']),json.dumps(data,ensure_ascii=False,allow_nan=False)))

    @staticmethod
    def all(db,kind,project=None):
        sql,args = 'SELECT data FROM entities WHERE kind=?',[kind]
        if project is not None:
            sql += ' AND project=?'
            args.append(project)
        sql += ' ORDER BY rowid'
        return [json.loads(r[0]) for r in db.execute(sql,args)]
