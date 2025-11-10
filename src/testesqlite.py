import sqlite3

# Conectar ao banco
conn = sqlite3.connect("db.sqlite")

# Criar um cursor (objeto que executa os comandos SQL)
cursor = conn.cursor()

# Ver todas as tabelas existentes
#cursor.execute("SELECT name FROM sqlite_master WHERE type='table';") #para ver as tabelas existentes


#para contar o total de linhas
cursor.execute("SELECT COUNT(*) FROM csat_extract;")
print(cursor.fetchone())