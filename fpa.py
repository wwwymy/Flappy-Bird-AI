import pygame
import random
# 导入neat库，用于实现神经网络和遗传算法
import neat
import os
import pygame.font


pygame.init()
WIDTH = 800
HEIGHT = 800


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Flappy Bird')
bg = pygame.image.load('img/bg.png')
ground = pygame.image.load('img/ground.png')
font = pygame.font.SysFont('Arial', 36)

class Bird:
    def __init__(self, x, y):
        self.images = []
        self.index = 0
        self.counter = 0
        for num in range(1, 4):
            img = pygame.image.load(f'img/bird{num}.png')
            self.images.append(img)
        self.image = self.images[self.index]
        self.x = x
        self.y = y
        self.v = 0
        # 标记是否点击
        self.clicked = False
    def jump(self):
        self.v = -10
        # 标记鸟已经跳跃
        self.clicked = True
        self.y += -33
    def move(self):
        self.v += 0.5
        if self.v > 10:
            self.v = 10
        if self.y < 650:
            self.y += int(self.v)
        # 拍打翅膀的效果
        self.counter += 1
        if self.counter > 13:
            self.counter = 0
            self.index += 1
            if self.index >= 3:
                self.index = 0
        self.image = self.images[self.index]
        # 图像旋转
        self.image = pygame.transform.rotate(self.images[self.index], self.v * -3) 
    
    def collide(self, pipe):
        # 掩码可以将图像中的非透明部分转换为可检测碰撞的区域。
        # 创建鸟的掩码
        bird_mask = pygame.mask.from_surface(self.image)
        # 创建上方管道的掩码
        pipe_mask_up = pygame.mask.from_surface(pipe.pipe_up)
        # 创建下方管道的掩码
        pipe_mask_down = pygame.mask.from_surface(pipe.pipe_down)
        # 计算上方管道掩码的偏移量
        offset_up = (pipe.rect_up.x - self.x, pipe.rect_up.y - round(self.y))
        # 计算下方管道掩码的偏移量
        offset_down = (pipe.rect_down.x - self.x, pipe.rect_down.y - round(self.y))
        point_up = bird_mask.overlap(pipe_mask_up, offset_up)
        point_down = bird_mask.overlap(pipe_mask_down, offset_down)
        if point_up or point_down:
            return True
        return False

class Pipe:
    def __init__(self, x, y):
        gap = 180
        self.x = x
        self.pipe_up = pygame.image.load('img/up.png')
        self.pipe_down = pygame.image.load('img/down.png')
        self.rect_up = self.pipe_up.get_rect(bottomleft=(x, y - gap // 2))
        self.rect_down = self.pipe_down.get_rect(topleft=(x, y + gap // 2)) 
        self.top = self.rect_up.bottom
        self.bottom = self.rect_down.top
        self.passed = False
        self.active = True  # 标记管道是否处于活动状态

    def move(self):
        if self.active:
            self.rect_up.x -= ground_v
            self.rect_down.x -= ground_v
            self.x -= ground_v
            if self.rect_up.right < 0:
                self.active = False
    
    def draw(self, screen):
        if self.active:
            screen.blit(self.pipe_up, self.rect_up)
            screen.blit(self.pipe_down, self.rect_down)
    

def draw_win(screen, birds, pipes,score):
    screen.blit(bg, (0, 0))
    for pipe in pipes:
        pipe.draw(screen)
    screen.blit(ground, (ground_d, 650))
    for bird in birds:
        screen.blit(bird.image, (bird.x, bird.y))
    
     # score
    scoretxt = font.render("Score: " + str(score), 1, (255, 255, 255))
    screen.blit(scoretxt, (10, 10))
    pygame.display.update()

def main(genomes, config):
    global score,n
    score = 0
    global screen
    nets = []
    birds = []
    ge = []
    
    for genome_id, genome in genomes:
        # 将初始适应度设为0
        genome.fitness = 0
        # 使用基因组和配置对象创建神经网络
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(100, HEIGHT // 2))
         # 将基因组对象添加到基因组列表中
        ge.append(genome)
    
    pipes = []
    pipe_freq = 2500
    last_pipe = pygame.time.get_ticks() - pipe_freq
    over = False
    global ground_d
    ground_d = 0
    global ground_v
    ground_v = 3
    
    # 设置一个时钟对象，来控制程序的帧率
    clock = pygame.time.Clock()
    while not over:
        # 设置每秒60帧
        clock.tick(60)
        ground_d -= ground_v
        if abs(ground_d) > 38:
            ground_d = 0
        current_time = pygame.time.get_ticks()

        if current_time - last_pipe >= pipe_freq:
            pipe_height = random.randint(130, 480)
            pipes.append(Pipe(WIDTH, pipe_height))
            last_pipe = current_time
        
        pipe_ind = 0
        if len(birds) > 0:
             # 确定神经网络输入中使用哪个管道（第一个或第二个）
            if len(pipes) > 1 and birds[0].x > pipes[0].rect_up.right:
                pipe_ind = 1
        if len(birds) == 0:
            print(score)
            break
        
        for x, bird in enumerate(birds):
             # 每帧增加鸟的适应度
            ge[x].fitness += 0.1
            bird.move()
            # 输入鸟坐标，管道口坐标
            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].top), abs(bird.y - pipes[pipe_ind].bottom)))
            # nets[birds.index(bird)].activate(...) 返回的是一个包含输出节点值的列表，而不是单个值
            # 用tanh,结果在-1，1 
            if output[0] > 0.5:
                bird.jump()
        
        # 检查管道碰撞
        for pipe in pipes:
            pipe.move()
            for bird in birds:
                if bird.collide(pipe):
                    ge[birds.index(bird)].fitness -= 3
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))
            
                # 检查是否通过了管道
                if not pipe.passed and pipe.rect_up.x + 90 < bird.x:
                    pipe.passed = True
                    score += 1
                    pipes.remove(pipe)
                    for genome in ge:
                        genome.fitness += 4
            
            
        
        #检查上下碰撞
        for bird in birds:
            if bird.y >= 650 or bird.y <= 0:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                over = True
                pygame.quit()
                quit()
                

        draw_win(screen, birds, pipes,score)


def run(config_file):
    # 加载 NEAT 的配置文件
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,config_file)
    # 创建一个 Population 对象作为 NEAT 运行的顶层对象
    p = neat.Population(config)
    p.run(main, 20)

if __name__ == '__main__':
    # 确定配置文件的路径。这里使用了路径操作，以确保脚本能在任何工作目录下成功运行。
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
     # 运行 NEAT 算法进行训练
    run(config_path)

