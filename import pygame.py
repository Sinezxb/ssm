import pygame
import random
import sys
import math

# --- 初始化设置 ---
pygame.init()

# 屏幕设置
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("雷霆战机：陨石危机")

# 鼠标锁定在窗口内
pygame.event.set_grab(True)

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (220, 20, 60)      # 敌机深红 - 普通
PLAYER_COLOR = (169, 169, 169) # 玩家战机灰
BULLET_YELLOW = (255, 255, 0)
BULLET_TAIL_YELLOW = (255, 255, 150)
ENEMY_BULLET_RED = (255, 69, 0)
ENEMY_BULLET_TAIL = (255, 140, 0)
ROCK_COLOR = (139, 69, 19) # 陨石褐色

# 不同类型敌机颜色
BIG_ENEMY_COLOR = (138, 43, 226)    # 紫色，大型
NORMAL_ENEMY_COLOR = RED            # 深红，中型
SMALL_ENEMY_COLOR = (30, 144, 255)  # 蓝色，小型

# 游戏常量
FPS = 60
PLAYER_MAX_HEALTH = 100         # 血条满值用（真实血量可以 >100）
COLLISION_DAMAGE = 10
KILL_HEAL_AMOUNT = 5

# Boss 常量
BOSS_MAX_HEALTH = 3000          # Boss 血量
BOSS_SCORE_BONUS = 500          # 击杀 Boss 额外加分

# 护盾常量
SHIELD_DURATION_MS = 3000       # 护盾 3 秒
SMALL_ENEMY_SHIELD_PROB = 0.3   # 小型机携带护盾概率

# 攻击增益常量
ATTACK_BUFF_DURATION_MS = 3000        # 攻击增益 3 秒
SMALL_ENEMY_ATTACK_PROB = 0.2         # 小型机携带攻击增益概率

clock = pygame.time.Clock()
font_name = pygame.font.match_font('arial')

# --- 辅助函数 ---

def draw_text(surf, text, size, x, y):
    try:
        font = pygame.font.Font(font_name, size)
    except:
        font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)

def draw_health_bar(surf, x, y, hp):
    # 血条显示 0~100，真实血量可以 >100
    if hp < 0:
        hp = 0
    if hp > PLAYER_MAX_HEALTH:
        hp = PLAYER_MAX_HEALTH
    BAR_LENGTH = 100
    BAR_HEIGHT = 10
    fill = (hp / PLAYER_MAX_HEALTH) * BAR_LENGTH
    outline_rect = pygame.Rect(x, y, BAR_LENGTH, BAR_HEIGHT)
    fill_rect = pygame.Rect(x, y, fill, BAR_HEIGHT)
    pygame.draw.rect(surf, RED, outline_rect)
    pygame.draw.rect(surf, GREEN, fill_rect)
    pygame.draw.rect(surf, WHITE, outline_rect, 2)

def draw_boss_health_bar(surf, boss_hp, boss_max_hp):
    if boss_hp < 0:
        boss_hp = 0
    if boss_hp > boss_max_hp:
        boss_hp = boss_max_hp
    BAR_LENGTH = 400
    BAR_HEIGHT = 15
    x = (SCREEN_WIDTH - BAR_LENGTH) // 2
    y = 40
    fill = (boss_hp / boss_max_hp) * BAR_LENGTH
    outline_rect = pygame.Rect(x, y, BAR_LENGTH, BAR_HEIGHT)
    fill_rect = pygame.Rect(x, y, fill, BAR_HEIGHT)
    pygame.draw.rect(surf, (139, 0, 0), outline_rect)    # 深红底
    pygame.draw.rect(surf, (255, 0, 0), fill_rect)       # 红色填充
    pygame.draw.rect(surf, WHITE, outline_rect, 2)
    draw_text(surf, "BOSS HP", 18, SCREEN_WIDTH // 2, y - 22)

# --- 类定义 ---

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.base_image = pygame.Surface((50, 40))
        self.base_image.set_colorkey(BLACK)
        # 玩家战机：机头朝上
        pygame.draw.polygon(self.base_image, PLAYER_COLOR, [(25, 0), (0, 35), (50, 35)]) # 机身
        pygame.draw.rect(self.base_image, PLAYER_COLOR, (20, 30, 10, 10)) # 尾喷
        pygame.draw.circle(self.base_image, RED, (25, 20), 5) # 驾驶舱

        self.image = self.base_image.copy()
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - 10
        self.health = 100     # 初始 100，无上限
        self.last_shot = pygame.time.get_ticks()
        self.shoot_delay = 200

        # 护盾状态 + 次数
        self.shield_active = False
        self.shield_end_time = 0
        self.shield_charges = 0

        # 攻击增益状态 + 次数
        self.attack_buff_active = False
        self.attack_buff_end_time = 0
        self.attack_buff_charges = 0

    def update(self):
        mouse_x, _ = pygame.mouse.get_pos()
        self.rect.centerx = mouse_x
        if self.rect.right > SCREEN_WIDTH: self.rect.right = SCREEN_WIDTH
        if self.rect.left < 0: self.rect.left = 0

        now = pygame.time.get_ticks()

        # 护盾计时
        if self.shield_active and now >= self.shield_end_time:
            self.shield_active = False

        # 攻击增益计时
        if self.attack_buff_active and now >= self.attack_buff_end_time:
            self.attack_buff_active = False

    def shoot(self):
        now = pygame.time.get_ticks()
        if now - self.last_shot > self.shoot_delay:
            self.last_shot = now
            if self.attack_buff_active:
                # 火力增益：发射两发子弹（左右各一发）
                bullet1 = Bullet(self.rect.centerx - 8, self.rect.top)
                bullet2 = Bullet(self.rect.centerx + 8, self.rect.top)
                all_sprites.add(bullet1, bullet2)
                bullets.add(bullet1, bullet2)
            else:
                bullet = Bullet(self.rect.centerx, self.rect.top)
                all_sprites.add(bullet)
                bullets.add(bullet)

    # 这些是“真正执行 buff” 的函数
    def _start_shield(self, duration_ms=SHIELD_DURATION_MS):
        now = pygame.time.get_ticks()
        if self.shield_active and self.shield_end_time > now:
            self.shield_end_time += duration_ms
        else:
            self.shield_active = True
            self.shield_end_time = now + duration_ms

    def _start_attack_buff(self, duration_ms=ATTACK_BUFF_DURATION_MS):
        now = pygame.time.get_ticks()
        if self.attack_buff_active and self.attack_buff_end_time > now:
            self.attack_buff_end_time += duration_ms
        else:
            self.attack_buff_active = True
            self.attack_buff_end_time = now + duration_ms

    # 捡到道具时：增加次数
    def gain_shield_charge(self):
        self.shield_charges += 1

    def gain_attack_charge(self):
        self.attack_buff_charges += 1

    # 玩家按键释放
    def use_shield(self):
        if self.shield_charges > 0:
            self.shield_charges -= 1
            self._start_shield()

    def use_attack_buff(self):
        if self.attack_buff_charges > 0:
            self.attack_buff_charges -= 1
            self._start_attack_buff()

class EnemyBullet(pygame.sprite.Sprite):
    """普通敌人子弹 - 带拖尾，支持横向速度"""
    def __init__(self, x, y, speed_y=7, speed_x=0, damage=COLLISION_DAMAGE):
        super().__init__()
        self.image = pygame.Surface((8, 25))
        self.image.set_colorkey(BLACK)
        pygame.draw.rect(self.image, ENEMY_BULLET_TAIL, (2, 0, 4, 20))   # 拖尾
        pygame.draw.circle(self.image, ENEMY_BULLET_RED, (4, 22), 4)     # 弹头
        
        self.rect = self.image.get_rect()
        self.rect.top = y
        self.rect.centerx = x
        self.speed_y = speed_y
        self.speed_x = speed_x
        self.damage = damage

    def update(self):
        self.rect.y += self.speed_y
        self.rect.x += self.speed_x
        if self.rect.top > SCREEN_HEIGHT or self.rect.right < 0 or self.rect.left > SCREEN_WIDTH:
            self.kill()

class BossCenterBullet(pygame.sprite.Sprite):
    """Boss 中央大炮子弹：体型大，范围广，可横向偏移"""
    def __init__(self, x, y, speed_y=12, speed_x=0, damage=500):
        super().__init__()
        self.image = pygame.Surface((28, 50))
        self.image.set_colorkey(BLACK)
        # 大号能量弹
        pygame.draw.ellipse(self.image, (255, 215, 0), (0, 15, 28, 35))  # 金色主体
        pygame.draw.ellipse(self.image, (255, 69, 0), (4, 5, 20, 25))    # 橙色核心
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.top = y
        self.speed_y = speed_y
        self.speed_x = speed_x
        self.damage = damage

    def update(self):
        self.rect.y += self.speed_y
        self.rect.x += self.speed_x
        if self.rect.top > SCREEN_HEIGHT or self.rect.right < 0 or self.rect.left > SCREEN_WIDTH:
            self.kill()

class EnemyPlaneBase(pygame.sprite.Sprite):
    """敌机基类：负责移动、边界处理，子类决定外观 & 射击模式"""
    def __init__(self):
        super().__init__()
        self.can_shoot = False
        self.shoot_prob = 0.0  # 每帧开火概率
        self.health = 1        # 默认一枪死

    def update(self):
        self.rect.y += self.speed_y
        self.rect.x += self.speed_x

        # 出屏幕则删除
        if (self.rect.top > SCREEN_HEIGHT or 
            self.rect.left < -50 or 
            self.rect.right > SCREEN_WIDTH + 50):
            self.kill()
            return

        # 射击
        if self.can_shoot and random.random() < self.shoot_prob:
            self.shoot()

    def shoot(self):
        pass

class BigEnemyPlane(EnemyPlaneBase):
    """大型敌机：体型大，慢速，多血，发射 3-5 列交叉火力"""
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((80, 60))
        self.image.set_colorkey(BLACK)
        pygame.draw.polygon(self.image, BIG_ENEMY_COLOR, [(40, 60), (5, 10), (75, 10)])
        pygame.draw.rect(self.image, BIG_ENEMY_COLOR, (30, 0, 20, 18))
        pygame.draw.circle(self.image, (75, 0, 130), (40, 30), 8)

        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0, SCREEN_WIDTH - self.rect.width)
        self.rect.y = random.randrange(-150, -60)
        self.speed_y = random.randrange(1, 3)
        self.speed_x = random.randrange(-1, 2)

        self.can_shoot = True
        self.shoot_prob = 0.005
        self.health = 8

    def shoot(self):
        num = random.randint(3, 5)
        center_x = self.rect.centerx
        start_y = self.rect.bottom
        for i in range(num):
            t = i - (num - 1) / 2
            speed_x = t * 1.5
            bullet = EnemyBullet(center_x, start_y, speed_y=6, speed_x=speed_x, damage=COLLISION_DAMAGE)
            all_sprites.add(bullet)
            enemy_bullets.add(bullet)

class NormalEnemyPlane(EnemyPlaneBase):
    """普通敌机：中等体型，单发子弹"""
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 40))
        self.image.set_colorkey(BLACK)
        pygame.draw.polygon(self.image, NORMAL_ENEMY_COLOR, [(25, 40), (0, 5), (50, 5)])
        pygame.draw.rect(self.image, NORMAL_ENEMY_COLOR, (20, 0, 10, 10))
        pygame.draw.circle(self.image, (50, 0, 0), (25, 20), 5)

        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0, SCREEN_WIDTH - self.rect.width)
        self.rect.y = random.randrange(-100, -40)
        self.speed_y = random.randrange(2, 6)
        self.speed_x = random.randrange(-1, 2)

        self.can_shoot = True
        self.shoot_prob = 0.01
        self.health = 1

    def shoot(self):
        bullet = EnemyBullet(self.rect.centerx, self.rect.bottom, speed_y=7, speed_x=0, damage=COLLISION_DAMAGE)
        all_sprites.add(bullet)
        enemy_bullets.add(bullet)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class Selfattention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super(Selfattention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = self.head_dim ** -0.5
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q * self.scale
        attn_weights = torch.einsum("bshd,bshd->bhs", q, k.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum("bhs,bshd->bshd", attn_weights, v)
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)

        output = self.out_proj(attn_output)
        return output
class SmallEnemyPlane(EnemyPlaneBase):
    """小型敌机：体型小，不会发射子弹，速度略快，可能携带护盾或攻击增益"""
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((32, 26))
        self.image.set_colorkey(BLACK)
        pygame.draw.polygon(self.image, SMALL_ENEMY_COLOR, [(16, 26), (0, 4), (32, 4)])
        pygame.draw.rect(self.image, SMALL_ENEMY_COLOR, (13, 0, 6, 6))
        pygame.draw.circle(self.image, (0, 0, 80), (16, 14), 4)

        # 是否携带护盾技能
        self.has_shield_powerup = (random.random() < SMALL_ENEMY_SHIELD_PROB)
        # 是否携带攻击增益
        self.has_attack_powerup = (random.random() < SMALL_ENEMY_ATTACK_PROB)

        # 视觉标记：护盾用淡蓝圈，攻击增益用橙圈，如果两种都有就两层圈
        if self.has_shield_powerup:
            pygame.draw.circle(self.image, (135, 206, 250), (16, 13), 15, 2)
        if self.has_attack_powerup:
            pygame.draw.circle(self.image, (255, 165, 0), (16, 13), 11, 2)

        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0, SCREEN_WIDTH - self.rect.width)
        self.rect.y = random.randrange(-80, -30)
        self.speed_y = random.randrange(3, 7)
        self.speed_x = random.randrange(-2, 3)

        self.can_shoot = False
        self.health = 1

class Rock(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        size = random.randrange(30, 60)
        self.image = pygame.Surface((size, size))
        self.image.set_colorkey(BLACK)
        
        points = []
        center = (size // 2, size // 2)
        radius = size // 2
        num_points = random.randint(5, 8)
        for i in range(num_points):
            angle = math.radians(i * (360 / num_points))
            r = radius * random.uniform(0.6, 1.0)
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            points.append((x, y))
        pygame.draw.polygon(self.image, ROCK_COLOR, points)
        
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0, SCREEN_WIDTH - self.rect.width)
        self.rect.y = random.randrange(-100, -40)
        self.speed_y = random.randrange(2, 5)
        self.speed_x = random.uniform(-0.5, 0.5)

    def update(self):
        self.rect.y += self.speed_y
        self.rect.x += self.speed_x
        if self.rect.top > SCREEN_HEIGHT:
            self.kill()

class Bullet(pygame.sprite.Sprite):
    """玩家子弹 - 带拖尾"""
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((6, 25))
        self.image.set_colorkey(BLACK)
        pygame.draw.rect(self.image, BULLET_TAIL_YELLOW, (1, 5, 4, 20))  # 拖尾
        pygame.draw.rect(self.image, BULLET_YELLOW, (0, 0, 6, 8))        # 弹头
        
        self.rect = self.image.get_rect()
        self.rect.bottom = y
        self.rect.centerx = x
        self.speed_y = -12

    def update(self):
        self.rect.y += self.speed_y
        if self.rect.bottom < 0:
            self.kill()

class Boss(pygame.sprite.Sprite):
    """终极 Boss：顶部巨大战舰
       攻击循环：左右同时 7 发散射 → 中央蓄力 → 大范围高速弹幕
    """
    def __init__(self):
        super().__init__()
        width = 300
        height = 120
        self.image = pygame.Surface((width, height))
        self.image.set_colorkey(BLACK)

        # 主体：大块舰体，压迫感
        pygame.draw.rect(self.image, (60, 60, 60), (20, 40, width - 40, 60))     # 主舰身
        pygame.draw.rect(self.image, (90, 90, 90), (60, 20, width - 120, 40))    # 上层结构
        pygame.draw.rect(self.image, (120, 0, 0), (0, 70, width, 20))            # 红色装甲带

        # 左右炮台基座
        pygame.draw.rect(self.image, (80, 80, 80), (40, 30, 30, 30))   # 左炮基座
        pygame.draw.rect(self.image, (80, 80, 80), (width - 70, 30, 30, 30))  # 右炮基座
        # 中央炮基座
        pygame.draw.rect(self.image, (100, 100, 100), (width//2 - 25, 10, 50, 40))

        # 炮管(画在图像上用于视觉，真实发射点用坐标计算)
        pygame.draw.rect(self.image, (200, 200, 200), (50, 0, 10, 30))  # 左炮管
        pygame.draw.rect(self.image, (200, 200, 200), (width - 60, 0, 10, 30)) # 右炮管
        pygame.draw.rect(self.image, (220, 220, 220), (width//2 - 8, 0, 16, 25)) # 中央短炮(外观)

        # 一些灯光点缀
        for x in range(40, width - 40, 30):
            pygame.draw.circle(self.image, (0, 255, 0), (x, 95), 4)

        self.rect = self.image.get_rect()
        self.rect.midtop = (SCREEN_WIDTH // 2, 0)
        self.speed_x = 2    # 左右移动

        # 血量
        self.health = BOSS_MAX_HEALTH

        # 攻击节奏：
        # phase 0: 左右炮同时散射
        # phase 1: 中央蓄力 -> 大范围弹幕
        self.phase = 0
        self.last_fire_time = pygame.time.get_ticks()
        self.side_delay = 1200              # 两侧攻击间隔
        self.center_delay = 2000            # 中央炮完整周期总间隔（包括蓄力）
        self.center_charge_time = 800       # 中央炮蓄力时间
        self.center_charging = False
        self.charge_start_time = 0

    def update(self):
        # 水平来回移动
        self.rect.x += self.speed_x
        if self.rect.left < 0 or self.rect.right > SCREEN_WIDTH:
            self.speed_x *= -1

        now = pygame.time.get_ticks()

        # 攻击状态机
        if self.phase == 0:
            # 左右同时散射
            if now - self.last_fire_time >= self.side_delay:
                self.fire_sides()
                self.last_fire_time = now
                # 进入中央蓄力阶段
                self.phase = 1
                self.center_charging = True
                self.charge_start_time = now
        elif self.phase == 1:
            # 中央蓄力阶段
            if self.center_charging and now - self.charge_start_time >= self.center_charge_time:
                self.fire_center()
                self.center_charging = False
                self.last_fire_time = now
                self.phase = 0  # 回到两侧攻击

    def _left_cannon_pos(self):
        x = self.rect.left + 50
        y = self.rect.top + 30
        return x, y

    def _right_cannon_pos(self):
        x = self.rect.right - 50
        y = self.rect.top + 30
        return x, y

    def _center_cannon_pos(self):
        x = self.rect.centerx
        y = self.rect.top + 25
        return x, y

    def fire_sides(self):
        # 左右两侧同时 7 发散射，形成交叉火力网
        left_x, left_y = self._left_cannon_pos()
        right_x, right_y = self._right_cannon_pos()

        patterns = [-3, -2, -1, 0, 1, 2, 3]  # 横向速度
        for sx in patterns:
            # 左边：向右偏 & 中心
            b_left = EnemyBullet(left_x, left_y, speed_y=7, speed_x=sx, damage=200)
            all_sprites.add(b_left)
            enemy_bullets.add(b_left)
            # 右边：镜像，向左偏 & 中心
            b_right = EnemyBullet(right_x, right_y, speed_y=7, speed_x=-sx, damage=200)
            all_sprites.add(b_right)
            enemy_bullets.add(b_right)

    def fire_center(self):
        # 中央大范围高速弹幕：多发大弹，扇形扫射
        cx, cy = self._center_cannon_pos()
        # 覆盖更大范围：横向速度从 -6 到 6
        patterns = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        for sx in patterns:
            b = BossCenterBullet(cx, cy, speed_y=12, speed_x=sx, damage=500)
            all_sprites.add(b)
            enemy_bullets.add(b)

# --- 游戏主程序 ---

all_sprites = pygame.sprite.Group()
mobs = pygame.sprite.Group()         # 敌机 + 陨石
enemies = pygame.sprite.Group()      # 敌机
bullets = pygame.sprite.Group()
enemy_bullets = pygame.sprite.Group()

player = Player()
all_sprites.add(player)

boss = None
boss_active = False
boss_spawned = False
boss_defeated = False

# 动态生成敌人的函数
def spawn_new_mob():
    if boss_active:
        return  # Boss 出现后不再生成新的飞机和岩石

    r = random.random()
    if r < 0.2:
        m = BigEnemyPlane()
    elif r < 0.7:
        m = NormalEnemyPlane()
    else:
        m = SmallEnemyPlane()
    enemies.add(m)
    all_sprites.add(m)
    mobs.add(m)

    # 30% 概率顺便生成一个陨石
    if random.random() < 0.3:
        rock = Rock()
        all_sprites.add(rock)
        mobs.add(rock)

# 初始生成一些敌人/陨石
for i in range(8):
    spawn_new_mob()

score = 0
pygame.mouse.set_visible(False)
running = True
game_over = False
win = False

while running:
    clock.tick(FPS)

    # 1. 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            # A 键：释放护盾
            elif event.key == pygame.K_a and not game_over:
                player.use_shield()
            # B 键：释放攻击增益
            elif event.key == pygame.K_b and not game_over:
                player.use_attack_buff()

    if not game_over:
        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_SPACE] or pygame.mouse.get_pressed()[0]:
            player.shoot()

        # 2. 更新
        all_sprites.update()

        # Boss 触发：得分到 1000 时
        if (not boss_spawned) and score >= 1000:
            boss_spawned = True
            boss_active = True
            # 清掉当前场上的小怪和陨石，只剩玩家 & 子弹
            for m in mobs:
                m.kill()
            # 生成 Boss
            boss = Boss()
            all_sprites.add(boss)

        # 小怪维护数量（只有在没有 Boss 时）
        if not boss_active:
            while len(mobs) < 12:
                spawn_new_mob()

        # 3. 碰撞检测

        # A. 玩家子弹击中 敌机 或 陨石（注意大型机要多次命中，小型机掉护盾/攻击增益）
        hits = pygame.sprite.groupcollide(mobs, bullets, False, True)
        for mob, hit_bullets in hits.items():
            num_hits = len(hit_bullets)

            if isinstance(mob, Rock):
                mob.kill()
                score += 10
                player.health += 2
                spawn_new_mob()

            elif isinstance(mob, BigEnemyPlane):
                mob.health -= num_hits
                if mob.health <= 0:
                    mob.kill()
                    score += 50
                    player.health += KILL_HEAL_AMOUNT * 2
                    spawn_new_mob()

            elif isinstance(mob, NormalEnemyPlane):
                mob.kill()
                score += 20
                player.health += KILL_HEAL_AMOUNT
                spawn_new_mob()

            elif isinstance(mob, SmallEnemyPlane):
                # 小型机：可能掉落护盾 & 攻击增益（变成可用次数）
                if getattr(mob, "has_shield_powerup", False):
                    player.gain_shield_charge()
                if getattr(mob, "has_attack_powerup", False):
                    player.gain_attack_charge()
                mob.kill()
                score += 15
                player.health += 3
                spawn_new_mob()

            else:
                mob.kill()
                score += 10
                player.health += KILL_HEAL_AMOUNT
                spawn_new_mob()

        # B. Boss 被玩家子弹击中
        if boss_active and boss is not None:
            boss_hits = pygame.sprite.spritecollide(boss, bullets, True)
            if boss_hits:
                boss.health -= 50 * len(boss_hits)
                if boss.health <= 0:
                    boss_active = False
                    boss_defeated = True
                    boss.kill()
                    score += BOSS_SCORE_BONUS
                    game_over = True
                    win = True

        # C. 敌机/陨石 撞到 玩家
        hits = pygame.sprite.spritecollide(player, mobs, True)
        for hit in hits:
            if not player.shield_active:
                player.health -= COLLISION_DAMAGE
            if not boss_active:
                spawn_new_mob()
            if player.health <= 0:
                game_over = True
                win = False

        # D. 敌人子弹击中玩家（包括 Boss 子弹，按 damage 结算）
        hits = pygame.sprite.spritecollide(player, enemy_bullets, True)
        for bullet in hits:
            if player.shield_active:
                # 有护盾时完全无视伤害
                continue
            dmg = getattr(bullet, "damage", COLLISION_DAMAGE)
            player.health -= dmg
            if player.health <= 0:
                game_over = True
                win = False

    # 4. 绘制
    screen.fill(BLACK)
    all_sprites.draw(screen)

    # Boss 中央蓄力特效（发射前一段时间高亮）
    if boss_active and boss is not None and boss.center_charging:
        cx, cy = boss._center_cannon_pos()
        # 简单的蓄力光圈效果
        pygame.draw.circle(screen, (255, 215, 0), (cx, cy), 30, 3)
        pygame.draw.circle(screen, (255, 140, 0), (cx, cy), 20, 2)

    # 玩家护盾特效（外层蓝圈）
    if player.shield_active:
        pygame.draw.circle(screen, (0, 191, 255), player.rect.center, 35, 2)
    # 攻击增益特效（内层金黄色圈）
    if player.attack_buff_active:
        pygame.draw.circle(screen, (255, 215, 0), player.rect.center, 28, 2)

    # UI绘制
    draw_text(screen, f"Score: {score}", 24, SCREEN_WIDTH / 2, 10)
    draw_text(screen, "Health:", 18, 40, 10)
    draw_health_bar(screen, 70, 15, player.health)

    # 显示 buff 次数
    draw_text(screen, f"Shield(A): {player.shield_charges}", 18, SCREEN_WIDTH - 120, 10)
    draw_text(screen, f"Power(B): {player.attack_buff_charges}", 18, SCREEN_WIDTH - 120, 35)

    # Boss 血条
    if boss_active and boss is not None:
        draw_boss_health_bar(screen, boss.health, BOSS_MAX_HEALTH)

    if game_over:
        if win:
            draw_text(screen, "YOU WIN!", 64, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 4)
        else:
            draw_text(screen, "GAME OVER", 64, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 4)
        draw_text(screen, f"Final Score: {score}", 30, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 50)
        draw_text(screen, "Press ESC to Quit", 22, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 20)
    
    pygame.display.flip()

pygame.quit()
sys.exit()
