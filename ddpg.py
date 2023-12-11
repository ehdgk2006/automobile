import torch
import torch.nn as nn

import numpy as np

from copy import deepcopy
from math import pi
import random

from replay_buffer import Transition, ReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, action_size),
            nn.Tanh()
        )
    

    def forward(self, x):
        z = self.model(x)
        
        return z


class Critic(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.in_features = state_size + action_size

        self.model = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 1),
            nn.Tanh()
        )
    

    def forward(self, x):
        z = torch.cat(x, 1)

        z = self.model(z)
        
        return z


class DDPG(nn.Module):
    def __init__(self, state_size: int, action_size: int, learning_rate: float, gamma: float, tau: float) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)

        self.target_actor.load_state_dict(deepcopy(self.actor.state_dict()))
        self.target_critic.load_state_dict(deepcopy(self.critic.state_dict()))

        self.critic_optimizer = torch.optim.SGD(params = self.critic.parameters(), lr = learning_rate)
        self.actor_optimizer = torch.optim.SGD(params = self.actor.parameters(), lr = learning_rate, maximize=True)
    

    def get_action(self, state, random_rate: float):
        with torch.no_grad():
            action = self.actor(state)
            r = 2*torch.rand_like(action) - 1
            return ((1 - random_rate) * action + random_rate * r)
    

    def optimize_critic(self, transition: list):
        batch_size = len(transition)
        batch = Transition(*zip(*transition))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        next_action_values = torch.zeros(batch_size, device=device).unsqueeze(1)
        with torch.no_grad():
            next_action_values[non_final_mask] = self.target_critic([non_final_next_states, self.target_actor(non_final_next_states)])
        expected_action_values = (next_action_values * self.gamma) + reward_batch

        action_values = self.critic([state_batch, action_batch])

        loss = nn.functional.mse_loss(action_values, expected_action_values)
        
        self.critic_optimizer.zero_grad()
        loss.backward()

        self.critic_optimizer.step()


    def optimize_actor(self, transition: list):
        batch = Transition(*zip(*transition))
        state_batch = torch.cat(batch.state)

        policy_grad = self.critic([state_batch, self.actor(state_batch)])
        policy_grad = torch.mean(policy_grad)

        self.actor_optimizer.zero_grad()
        policy_grad.backward()

        self.actor_optimizer.step()

    
    def update_target_net(self):
        for target_critic_param, critic_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_critic_param.data.copy_(self.tau*critic_param.data + (1.0-self.tau)*target_critic_param.data)
        
        for target_actor_param, actor_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_actor_param.data.copy_(self.tau*actor_param.data + (1.0-self.tau)*target_actor_param.data)
        

    def optimize(self, transition: list):
        self.optimize_critic(transition)
        self.optimize_actor(transition)

        self.update_target_net()


if __name__ == '__main__':
    from toy import Rectangle
    from raycast import Raycast
    from car import Car
    from world import World
    from vector import Vector
    from utils import draw

    from explorer import Trajectory, Cell, Archive

    import pygame
    import sys
    import threading


    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
    
    SCREEN_WIDTH = 256
    SCREEN_HEIGHT = 256

    FRAME = 30
    EPISODE = 9999999999
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    TAU = 0.001
    BATCH_SIZE = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDPG(10, 2, LEARNING_RATE, GAMMA, TAU).to(device)
    archive = Archive()
    buffer = ReplayBuffer(10000)


    def get_state(world: World):
        ray_check = world.ray_check()

        res = list(map(lambda r: float(len(r) != 0), ray_check))
        pos = (world.goal.pos - world.car.body.pos)
        pos.rotate(-world.car.direction_rad)
        pos /= SCREEN_WIDTH / 2.
        res += pos.v

        return res
    

    def explore(k):
        global goal, main_world, archive, buffer

        # init world
        main_world = init_world()

        # select cell
        selected_cell = deepcopy(archive.select_cell())

        # representate cell
        cur_state = selected_cell.trajectory.real_state
        if cur_state != None:
            main_world.car.rotate(cur_state[0])
            main_world.car.transform(Vector(cur_state[1], cur_state[2]))
            main_world.goal.pos = Vector(cur_state[3], cur_state[4])

        action = [random.random() * 2 - 1, random.random() * 2 - 1]
        for i in range(k):
            # random action (exploration)
            state = get_state(main_world)
            if random.random() <= 0.05:
                action = [random.random() * 2 - 1, random.random() * 2 - 1]
            main_world.car.set_handle(action[0] * main_world.car.max_angle)
            main_world.car.set_speed(action[1] * main_world.car.max_speed)
            main_world.car.update(1/FRAME)
            terminate = main_world.check_goal()
            reward = 0

            if len(main_world.collision_check()):
                reward = -1.

                t = Transition(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0),
                    None,
                    torch.tensor([reward], dtype=torch.float32, device=device).unsqueeze(0))
                buffer.push(t)

                main_world = init_world()
                rs = [main_world.car.direction_rad] + main_world.car.body.pos.v + main_world.goal.pos.v
                s = get_state(main_world)
                for i in range(len(s)):
                    s[i] = round(s[i], 2)
                
                archive.push(Cell(s, rs))
                return

            elif terminate:
                next_state = get_state(main_world)
                reward = 1.

                goal_x = random.randint(-SCREEN_WIDTH / 2 + 70, SCREEN_WIDTH / 2 - 70)
                goal_y = random.randint(-SCREEN_HEIGHT / 2 + 70, SCREEN_HEIGHT / 2 - 70)
                goal = Rectangle('goal', Vector(goal_x, goal_y), 20, 20, True)
                main_world.goal = goal

                w = init_world()
                w.car.rotate(selected_cell.trajectory.first_state[0])
                w.car.transform(Vector(selected_cell.trajectory.first_state[1], selected_cell.trajectory.first_state[2]))
                w.goal.pos = Vector(selected_cell.trajectory.first_state[3], selected_cell.trajectory.first_state[4])

                for i in range(len(selected_cell.trajectory)):
                    s = get_state(w)
                    dist1 = (w.car.body.pos - w.goal.pos).norm(2)
                    a = selected_cell.trajectory.trajectory[i]
                    w.car.set_handle(a[0] * main_world.car.max_angle)
                    w.car.set_speed(a[1] * main_world.car.max_speed)
                    w.car.update(1/FRAME)
                    ns = get_state(w)
                    dist2 = (w.car.body.pos - w.goal.pos).norm(2)

                    t = Transition(
                        torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0),
                        torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0),
                        torch.tensor(ns, dtype=torch.float32, device=device).unsqueeze(0),
                        torch.tensor([(float(dist1 > dist2) - 0.5) / 100.], dtype=torch.float32, device=device).unsqueeze(0))
                    buffer.push(t)
                
                t = Transition(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor([reward], dtype=torch.float32, device=device).unsqueeze(0))
                buffer.push(t)

                a = deepcopy(action)
                rs = [main_world.car.direction_rad] + main_world.car.body.pos.v + main_world.goal.pos.v
                s = get_state(main_world)
                for i in range(len(s)):
                    s[i] = round(s[i], 2)
                
                selected_cell.add_trajectory(a, deepcopy(s), deepcopy(rs), 0)
                if terminate:
                    selected_cell.trajectory.reset(deepcopy(s), deepcopy(rs))
                archive.see_cell(selected_cell)

                break
                    
            a = deepcopy(action)
            rs = [main_world.car.direction_rad] + main_world.car.body.pos.v + main_world.goal.pos.v
            s = get_state(main_world)
            for i in range(len(s)):
                s[i] = round(s[i], 2)
            
            selected_cell.add_trajectory(a, deepcopy(s), deepcopy(rs), 0)
            archive.see_cell(selected_cell)

        # push new cell
        archive.push(selected_cell)


    def init_world() -> World:
        car = Car(
            Rectangle('car', Vector(0, 0), 30, 50, True),
            [
                Raycast(Vector(0, 0), Vector(0, 0), Vector(40 * 1.414, 0)),     # E
                Raycast(Vector(0, 0), Vector(0, 0), Vector(40, 40)),            # NE
                Raycast(Vector(0, 0), Vector(0, 0), Vector(0, 40 * 1.414)),     # N
                Raycast(Vector(0, 0), Vector(0, 0), Vector(-40, 40)),           # NW
                Raycast(Vector(0, 0), Vector(0, 0), Vector(-40 * 1.414, 0)),    # W
                Raycast(Vector(0, 0), Vector(0, 0), Vector(40, -40)),           # SW
                Raycast(Vector(0, 0), Vector(0, 0), Vector(0, -40 * 1.414)),    # S
                Raycast(Vector(0, 0), Vector(0, 0), Vector(-40, -40)),          # SE
            ]
        )

        goal_x = random.randint(-SCREEN_WIDTH / 2 + 70, SCREEN_WIDTH / 2 - 70)
        goal_y = random.randint(-SCREEN_HEIGHT / 2 + 70, SCREEN_HEIGHT / 2 - 70)
        wall_up = Rectangle('wall', Vector(0, SCREEN_HEIGHT / 2 - 5), SCREEN_WIDTH, 10, True)
        wall_down = Rectangle('wall', Vector(0, -SCREEN_HEIGHT / 2 + 5), SCREEN_WIDTH, 10, True)
        wall_left = Rectangle('wall', Vector(-SCREEN_WIDTH / 2 + 5, 0), 10, SCREEN_HEIGHT, True)
        wall_right = Rectangle('wall', Vector(SCREEN_WIDTH / 2 - 5, 0), 10, SCREEN_HEIGHT, True)

        goal = Rectangle('goal', Vector(goal_x, goal_y), 20, 20, True)

        return World(car, goal, [wall_up, wall_down, wall_left, wall_right])

    
    def gui():
        white = (255, 255, 255)
        green = (0, 255, 0)
        red = (255, 0, 0)
        black = (0, 0, 0)

        pygame.init()
        pygame.display.set_caption("DDPG automobile")
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        clock = pygame.time.Clock()

        while True:
            clock.tick(FRAME)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            screen.fill(black)
            draw(screen, white, main_world.car.body.pos,  main_world.car.body.points)
            draw(screen, green, main_world.goal.pos, main_world.goal.points)
            
            l = main_world.ray_check()

            for i, ray in enumerate(main_world.car.rays):
                if len(l[i]) == 0:
                    draw(screen, green, main_world.car.body.pos, [ray.ray_start, ray.ray_end])
                else:
                    draw(screen, red, main_world.car.body.pos, [ray.ray_start, ray.ray_end])
            for toy in main_world.toys:
                draw(screen, white, toy.pos, toy.points)
            pygame.display.update()
    

    t = threading.Thread(target=gui)
    t.start()

    main_world = init_world()
    s = get_state(main_world)
    for i in range(len(s)):
        s[i] = round(s[i], 2)

    archive.push(Cell(s, get_state(main_world)))
    for episode in range(EPISODE):
        explore(100)

        if len(buffer) < BATCH_SIZE:
            continue

        model.optimize(buffer.sample(BATCH_SIZE))
        
        if episode % 100 == 0:
            torch.save(model.state_dict(), "./automobile.pth")
            print(f"episode: {episode}")
            print(f"\tstate: {buffer[-1].state.cpu().numpy()} \n\t action: {buffer[-1].action.cpu().numpy()}\n\t reward: {buffer[-1].reward.cpu().numpy()}")

