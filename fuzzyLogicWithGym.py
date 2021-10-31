import numpy as np
import skfuzzy as fuzz
import matplotlib
from skfuzzy import control as ctrl
import gym
env = gym.make('MountainCarContinuous-v0')

print(env.action_space, env.action_space.low, env.action_space.high)
print(env.observation_space,env.observation_space.low, env.observation_space.high)

dystans = ctrl.Antecedent(np.arange(-1.2,0.6,0.01), 'dystans')
predkosc = ctrl.Antecedent(np.arange(-0.07,0.07,0.01), 'predkosc')
przyspieszenie = ctrl.Consequent(np.arange(-1,1,0.01), 'przyspieszenie')

dystans.automf(3, names=['raz','dwa','trzy'])

przyspieszenie['silneLewo'] = fuzz.trimf(przyspieszenie.universe, [-1, -1, 0.5])
przyspieszenie['silnePrawo'] = fuzz.trimf(przyspieszenie.universe, [-0.5, 1, 1])

predkosc['lewo'] = fuzz.trimf(predkosc.universe,  [-0.07,-0.07, 0.01])
predkosc['prawo'] = fuzz.trimf(predkosc.universe, [-0.01, 0.07, 0.07])

rule1 = ctrl.Rule(dystans['raz'] & predkosc['lewo'], przyspieszenie['silneLewo'])
rule2 = ctrl.Rule(dystans['raz'] & predkosc['prawo'], przyspieszenie['silnePrawo'])
rule3 = ctrl.Rule(dystans['dwa'] & predkosc['lewo'], przyspieszenie['silneLewo'])
rule4 = ctrl.Rule(dystans['dwa'] & predkosc['prawo'], przyspieszenie['silnePrawo'])
rule5 = ctrl.Rule(dystans['trzy'] & predkosc['lewo'], przyspieszenie['silneLewo'])
rule6 = ctrl.Rule(dystans['trzy'] & predkosc['prawo'], przyspieszenie['silnePrawo'])

system_ctrl = ctrl.ControlSystem([rule1, rule3,rule5,rule2, rule4,rule6])
system = ctrl.ControlSystemSimulation(system_ctrl)
#przyspieszenie.view(sim=system)
observation = env.reset()

for t in range(1000):
	env.render()
	#print(observation)
	position, velocity = observation
	system.input['dystans'] = position
	system.input['predkosc'] = velocity
	system.compute()
	action = np.array([system.output['przyspieszenie']])
	#przyspieszenie.view(sim=system)
	observation, reward, done, info = env.step(action)
	if done:
		print(f"Episode finished after {t} time steps.")
		break

env.close()

