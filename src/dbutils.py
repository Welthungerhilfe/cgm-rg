#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Tristan Behrens <tristan@ai-guru.de> for Welthungerhilfe
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import json
import psycopg2.extras
import os
import config

def connect_to_default_database():
    """
    Connects to the database. Should only be used when you want to create a new database. Do not do anything else there. Uses credentials from the JSON file.
    """
    
    json_data = load_dbconnection_file()
    dbname="postgres"
    user=json_data["user"]
    host=json_data["host"]
    password=json_data["password"]
    port=json_data["port"]
    sslmode=json_data["sslmode"]
    return DatabaseInterface(dbname=dbname, user=user, host=host, password=password, port=port, sslmode=sslmode)
    
    
def connect_to_main_database():
    """
    Connect to the main database. Uses database name and credentials from the JSON file.
    """
    
    dbname=config.DB_NAME
    user=confdbname=config.DB_USER
    host=confdbname=config.DB_HOST
    password=confdbname=config.DB_PASSWD
    port=confdbname=config.DB_PORT
    sslmode=confdbname=config.DB_SSL_MODE
    return DatabaseInterface(dbname=dbname, user=user, host=host, password=password, port=port, sslmode=sslmode)
    
    
def load_dbconnection_file(connection_file):
    """
    Loads the JSON file.
    """
    #with open("/home/mmatiaschek/conf/dbconnection.json") as json_file:  
    with open(connection_file) as json_file:
        json_data = json.load(json_file)
        return json_data

    
class DatabaseInterface:
    
    def __init__(self, dbname, user, host, password, port, sslmode):
        """
        Established a connection to a database.
        """
        
        self.dbname = dbname
        self.user = user
        self.host = host
        self.password = password
        self.port = port
        self.sslmode = sslmode
        self.connect()
    
    def connect(self):
        """
        Connects.
        """
        
        self.connection = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            host=self.host,
            password=self.password,
            port=self.port,
            sslmode=self.sslmode
        )
        self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()
        self.dictionary_cursor = self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
     
    def execute_script_file(self, filename):
        """
        Executes a SQL script file.
        """
        
        script = open(filename, "r").read()
        self.execute(script)
        
        
    def execute(self, script, fetch_one=False, fetch_all=False, dict_cursor=False):
        """
        Executes an SQL file. Capable of reconnecting if necessary.
        """

        # Try to  execute the script.
        try:
            #result = self.cursor.execute(script)
            if dict_cursor:
                result = self.dictionary_cursor.execute(script)
            else:
                result = self.cursor.execute(script)

        # Could happen. Reconnect. And try again.
        except psycopg2.OperationalError:
            self.connect()
            #result = self.cursor.execute(script)
            if dict_cursor:
                result = self.dictionary_cursor.execute(script)
            else:
                result = self.cursor.execute(script)

        # Commit the statement.
        self.connection.commit()
        if fetch_one == True:
            #result = self.cursor.fetchone()
            if dict_cursor:
                result = self.dictionary_cursor.fetchone()
            else:
                result = self.cursor.fetchone()
        
        elif fetch_all == True:
            #result = self.cursor.fetchall()
            if dict_cursor:
                 result = self.dictionary_cursor.fetchall()
            else:
                result = self.cursor.fetchall()

        return result

    def batch_execute(self, artifact_df, table):
        df_columns = list(artifact_df)
        # create (col1,col2,...)
        columns = ",".join(df_columns)

        # create VALUES('%s', '%s",...) one '%s' per column
        values = "VALUES({})".format(",".join(["%s" for _ in df_columns]))

        #create INSERT INTO table (columns) VALUES('%s',...)
        insert_stmt = "INSERT INTO {} ({}) {}".format(table,columns,values)

        result = psycopg2.extras.execute_batch(self.cursor, insert_stmt, artifact_df.values)
        self.connection.commit()

        return result        
    
    def get_all_tables(self):
        """
        Retrieves all tables.
        """
        
        self.cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = [str(table[0]) for table in self.cursor.fetchall()]
        return tables
    
    def clear_table(self, table):
        result = self.cursor.execute("TRUNCATE TABLE {};".format(table))
        self.connection.commit()

    def get_number_of_rows(self, table):
        self.cursor.execute("SELECT COUNT(*) from {};".format(table))
        result = self.cursor.fetchone()
        return result[0]
    
    def get_columns(self, table, get_column_name=True, get_data_type=False):
        sql_statement = ""
        sql_statement += "SELECT column_name, data_type, character_maximum_length FROM INFORMATION_SCHEMA.COLUMNS" 
        sql_statement += " WHERE table_name = '{}';".format(table)
        self.cursor.execute(sql_statement)
        results = self.cursor.fetchall()
        if get_data_type:
            columns_info = [result[0:2] for result in results]
        else:
            columns_info = [result[0] for result in results]
        return columns_info    
        
        #columns = [result[0] for result in results]
        #return columns
    

def create_insert_statement(table, keys, values, convert_values_to_string=True, use_quotes_for_values=True):
    if convert_values_to_string == True:
        values = [str(value) for value in values]
    if use_quotes_for_values == True:
        values = ["'" + value + "'" for value in values]
        
    sql_statement = "INSERT INTO {}".format(table) + " "
    
    keys_string = "(" + ", ".join(keys) + ")"
    sql_statement += keys_string
    
    values_string = "VALUES (" + ", ".join(values) + ")"
    sql_statement += "\n" + values_string
    
    sql_statement += ";" + "\n"
    
    return sql_statement


def create_update_statement(table, keys, values, id_value, convert_values_to_string=True, use_quotes_for_values=True):
    if convert_values_to_string == True:
        values = [str(value) for value in values]
    if use_quotes_for_values == True:
        values = ["'" + value + "'" for value in values]
        
    Pkey = [value for key, value in zip(keys, values) if key == id_value]    
    
    sql_statement = "UPDATE {}".format(table) + " SET"
    sql_statement += ", ".join([" {} = {}".format(key, value) for key, value in zip(keys, values) if key != id_value])
    sql_statement += " WHERE id = {}".format(Pkey[0])
    sql_statement += ";" + "\n"

    return sql_statement


def create_select_statement(table, keys=[], values=[], convert_values_to_string=True, use_quotes_for_values=True):
    if convert_values_to_string == True:
        values = [str(value) for value in values]
    if use_quotes_for_values == True:
        values = ["'" + value + "'" for value in values]

    sql_statement = "SELECT * FROM {}".format(table)
    
    if len(keys) != 0 and len(values) != 0:
        sql_statement += " WHERE "
        like_statements = []
        for key, value in zip(keys, values):
            like_statement = str(key) + "=" + str(value)
            like_statements.append(like_statement)
        sql_statement += " AND ".join(like_statements) 
    
    sql_statement += ";" + "\n"
    return sql_statement