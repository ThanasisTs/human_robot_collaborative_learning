#!/usr/bin/env python
import rospy
import pygame as pg
from transfer_learning_SAC.msg import Score

class ShowScore():
    def __init__(self):
        sub = rospy.Subscriber('score_topic', Score, self.score_callback)
        pg.init()
        pg.font.init()
        self.display = (800, 800)
        self.screen = pg.display.set_mode(self.display)
        self.myFont = pg.font.SysFont('timesnewroman', 30)
        self.outcome = None
        self.msg_count, self.clean_screen_count = 0, 0
        self.running = True

    def run(self):
        text_titles = [self.myFont.render("Game Outcome", False, (255, 0, 0)),
                        self.myFont.render("Score", False, (255, 0, 0))]
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_q):
                    self.running = False
            rospy.sleep(0.01)
            if self.clean_screen_count != self.msg_count:
                self.clean_screen_count = self.msg_count
                self.screen.fill((0, 0, 0))
            self.screen.blit(text_titles[0], (self.display[0]//4 - text_titles[0].get_width()//2, 10))
            self.screen.blit(text_titles[1], (3*self.display[0]//4 - text_titles[1].get_width()//2, 10))
            if self.outcome is not None:
                text_episode = self.myFont.render("Episode: {}".format(self.msg_count), False, (255, 0, 0))
                text_episode_width = text_episode.get_width()
                self.screen.blit(text_episode, ((self.display[0]-text_episode_width)//2, 700))
                if self.outcome:
                    text_data = [self.myFont.render("WIN", False, (255, 0, 0)), self.myFont.render(str(self.score+150), False, (255, 0, 0))]
                else:
                    text_data = [self.myFont.render("LOSE", False, (255, 0, 0)), self.myFont.render(str(self.score+150), False, (255, 0, 0))]
                self.screen.blit(text_data[0], (self.display[0]//4 - text_data[0].get_width()//2, 90))
                self.screen.blit(text_data[1], (3*self.display[0]//4 - text_data[1].get_width()//2, 90))                
            pg.display.update()

    def score_callback(self, msg):
        self.msg_count += 1
        self.outcome = msg.outcome.data
        self.score = msg.score.data
    
def main():
    rospy.init_node("score_visualization")
    show_score = ShowScore()
    show_score.run()

if __name__ == "__main__":
    main()