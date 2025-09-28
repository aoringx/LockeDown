from ranking import Ranking
import random

class Bots:
    def __init__(self, rank, bots={}):
        self.bots = bots.keys()
        self.ranking_system = rank
        for bot in bots.keys():
            self.ranking_system.add_player(bot, bots[bot])

    def list_bots(self):
        return self.bots
    
    def change_bots(self):
        for bot in self.bots:
            self.ranking_system.increment_score(bot, (random.random()*2-1)/100)
    

if __name__ == "__main__":
    rank = Ranking()
    bots = Bots(rank,
                {"bot1":0,
                 "bot2":0.5,
                 "bot3":0.75})
    bots.change_bots()
    rank.close()