import sqlite3


# db= sqlite3.connect("db.sqlite3")

# db.execute("""CREATE TABLE IF NOT EXISTS 'config' ( 
        
#         `id` INTEGER PRIMARY KEY AUTOINCREMENT,
#         `key` varchar(30),
#         `value` varchar(255)
#     );""")

# db.commit()


# db.execute("""insert into `config` (`key`, `value`) VALUES (
#     'status','1'
#     ); """)

# db.commit()


class Settings:
    
    def __init__(self) -> None:
        self.db = sqlite3.connect("db.sqlite3")
        self.init()
    
    
    def init(self):
        self.db.execute("""CREATE TABLE IF NOT EXISTS 'config' ( 
        
            `id` INTEGER PRIMARY KEY AUTOINCREMENT,
            `key` varchar(30),
            `value` varchar(255)
        );""")
        
        self.db.commit()
        # self.db.execute("""insert into `config` (`key`, `value`) VALUES (
        #         'status','1'
        #     ); """)
        # self.db.commit()
        
    def bot_status(self):
        cursor = self.db.cursor()
        cursor.execute("select value from config where key='bot_status';")
        row=cursor.fetchone()
        if row is None :
            self.db.execute("""insert into `config` (`key`, `value`) VALUES (
                'bot_status','0'
            ); """)
            self.db.commit()
            cursor.execute("select value from config where key='bot_status';")
            row=cursor.fetchone()            # self.bot_status()

        return row[0]
    
    def turn_bot_on(self):
        if self.bot_status() !='1':
            cursor = self.db.cursor()
            cursor.execute("""update config set value='1' where key='bot_status'""")
            self.db.commit()
        return True
    
    def turn_bot_of(self):
        if self.bot_status() !='0':
            cursor = self.db.cursor()
            cursor.execute("""update config set value='0' where key='bot_status'""")
            self.db.commit()
        return True


    def trading_status(self):
        
        cursor = self.db.cursor()
        cursor.execute("select value from config where key='trading_status';")
        row=cursor.fetchone()
        if row is None :
            self.db.execute("""insert into `config` (`key`, `value`) VALUES (
                'trading_status','0'
            ); """)
            self.db.commit()
            cursor.execute("select value from config where key='trading_status';")
            row=cursor.fetchone()            # self.status()
        return row[0]
    
    def turn_trading_on(self):
        if self.trading_status() !='1':
            cursor = self.db.cursor()
            cursor.execute("""update config set value='1' where key='trading_status'""")
            self.db.commit()
        return True
    
    def turn_trading_of(self):
        if self.trading_status() !='0':
            cursor = self.db.cursor()
            cursor.execute("""update config set value='0' where key='trading_status'""")
            self.db.commit()
        return True



settings=Settings()


# status= settings.bot_status()
# print(status)
# settings.turn_bot_of()
# status= settings.bot_status()
# print(status)
# settings.turn_bot_on()
# status= settings.bot_status()
# print(status)


status= settings.trading_status()
print(status)
settings.turn_trading_of()
status= settings.trading_status()
print(status)
settings.turn_trading_on()
status= settings.trading_status()
print(status)
