I want you to build ONLY the weak scripted teacher first.

DO NOT:
- train anything
- modify the 1M checkpoint
- modify PPO
- modify the reward
- change the current 92D observation
- introduce RelativeDefaultObs yet
- build the BC pipeline yet
- build PPO fine-tuning yet
- change python-example gameplay
- add teacher fallback to the ML bot

The only objective right now is:

BUILD AND VALIDATE A WEAK SCRIPTED TEACHER THAT CAN RELIABLY APPROACH THE BALL AND TRY TO SEND IT TOWARD THE OPPONENT GOAL.

==================================================
WHY WE ARE DOING THIS
==================================================

Our current 1M PPO model learned some transferable ball-contact behavior in controlled RLGym/RocketSim fixtures, but it behaves poorly in live Rocket League.

Important evidence:

- Bridge 8000 argmax touch rate: 2.5% → 65%
- Touch 3000 argmax: 0% → 37.5%
- Approach 3000 argmax: 10% → 25%
- Standard kickoff-v2: 0/20 learner touches
- Successful contacts were overwhelmingly airborne
- The model can initiate forward propulsion
- The model jumps extremely early
- In live RLBot it often circles instead of approaching the ball
- Live logs showed repeated action families such as:
  action 51 = throttle +1, steer -1, yaw -1, boost 1
  action 12 = throttle +1, steer -1, yaw -1, boost 0
- I manually pushed the ball close to the bot and it STILL sometimes circled around instead of approaching properly
- The 1M policy sensitivity test showed that changing ball position does change the model's probabilities, but usually does not produce useful directional steering
- No confirmed feature-order, scaling, team inversion, or numeric action-table mismatch has been found

The current interpretation is that the PPO policy is brittle and/or learned a narrow behavior from a narrow training distribution.

Therefore we are considering:

WEAK TEACHER
→ BEHAVIOR CLONING
→ PPO FINE-TUNING

But before doing any of that, we must prove the teacher itself works.

==================================================
PROJECT
==================================================

Root:

C:\Users\shreyas\Desktop\model wars

Training:

C:\Users\shreyas\Desktop\model wars\training

Existing RLBot deployment:

C:\Users\shreyas\Desktop\model wars\python-example

Use the existing RLGym/RocketSim environment for the teacher.

Do NOT replace the current training system.

==================================================
TEACHER DESIGN
==================================================

The teacher is deliberately weak.

It should NOT try to be Nexto/Necto.

It should NOT implement:
- advanced aerial mechanics
- air dribbles
- flicks
- sophisticated defense
- wall play
- complex boost management
- advanced ball prediction
- complicated strategic positioning

The teacher's job is simply:

1. Find the ball.
2. Identify the opponent goal.
3. Determine the desired direction for the next ball touch.
4. Calculate a target point BEHIND the ball relative to the opponent goal.
5. Drive toward that target point.
6. Orient the car so that it can contact the ball usefully.
7. Approach the ball.
8. Make the ball move generally toward the opponent goal.
9. Use basic boost when appropriate.
10. Use simple jump behavior only if necessary, but do NOT force jumping.

Conceptually:

OPPONENT GOAL
      ↑
      |
desired ball travel direction
      |
  [TARGET POINT]
      |
     BALL
      |
     CAR

The target point should be on the side of the ball opposite the opponent goal, so that approaching through that point tends to push the ball toward the goal.

IMPORTANT:
The teacher should NOT simply do:

"drive directly at the ball."

It should attempt to approach the correct striking side of the ball.

==================================================
ACTION SPACE
==================================================

Use the SAME 90-action LookupTableAction space currently used by our training environment.

The teacher should output one LookupTableAction.

Do NOT create a continuous controller.

Do NOT introduce a new action space.

This ensures teacher demonstrations can later be used directly for behavior cloning.

==================================================
OBSERVATION / STATE
==================================================

For now, do NOT change the model observation.

The teacher can use whatever game-state information is necessary internally.

However, when generating demonstrations, save the SAME existing 92D DefaultObs that the eventual ML student would receive.

That means demonstration records should conceptually be:

92D observation
→ teacher LookupTableAction index

Do not introduce RelativeDefaultObs yet.

We will compare relative observations separately later.

==================================================
TEACHER BEHAVIOR DETAILS
==================================================

The teacher should reason about:

- car position
- car forward direction
- car velocity
- ball position
- ball velocity
- opponent goal position
- distance to target point
- angle to target point
- distance to ball
- whether car is grounded
- boost availability

For basic steering:

Calculate the target point behind the ball.

Determine the angle between the car's forward direction and the target point.

Use that angle to select steering/yaw direction.

Drive toward the target.

As the car gets close to the target/ball:
- reduce unnecessary steering
- avoid endlessly orbiting the ball
- avoid blindly accelerating past the ball
- attempt to align before contact

Use the simplest robust controller you can make.

Do NOT over-engineer it.

==================================================
CRITICAL: AVOID THE CIRCLE FAILURE
==================================================

Our current ML bot often gets into:

throttle
+
constant steering
+
constant yaw

and circles the ball.

The teacher MUST be tested specifically for this.

Add diagnostics such as:

- distance to target point
- distance to ball
- angle-to-target
- angle-to-ball
- current steering command
- target point
- car position
- ball position
- whether distance is decreasing

The teacher must not be considered successful merely because it moves.

==================================================
VALIDATION — DO NOT SKIP
==================================================

Before generating a large dataset, test the teacher.

Run controlled RocketSim scenarios.

Test at minimum:

1. Ball directly ahead.
2. Ball ahead-left.
3. Ball ahead-right.
4. Ball directly left.
5. Ball directly right.
6. Ball behind.
7. Ball close.
8. Ball medium distance.
9. Ball far.
10. Car initially facing away from the ball.
11. Ball moving slowly.
12. Ball moving laterally.
13. Ball moving toward opponent goal.
14. Ball moving away from opponent goal.

Use multiple starting positions.

For each test record:

- initial car position
- initial ball position
- initial ball velocity
- opponent goal
- teacher target point
- teacher actions over time
- distance to target over time
- distance to ball over time
- whether ground contact occurred
- whether ball contact occurred
- ball velocity after contact
- whether ball velocity has a component toward opponent goal
- whether the car enters a circular orbit around the ball

==================================================
SUCCESS CRITERIA
==================================================

Do NOT judge the teacher by "looks okay."

Measure it.

At minimum calculate:

1. Ball approach success:
Did car-to-ball distance decrease substantially?

2. Target approach:
Did car reach the target point behind the ball?

3. Grounded approach:
Did the car remain grounded during most of the approach?

4. Contact rate:
What percentage of episodes resulted in a ball contact?

5. Useful contact:
After contact, did the ball's velocity have a positive component toward the opponent goal?

6. Circling:
How often did the car orbit/circle the ball instead of converging?

7. Recovery:
After overshooting or missing, can the teacher recover rather than remain in a loop?

8. Time to contact.

9. Miss rate.

10. Behavior across different starting geometries.

==================================================
IMPORTANT TEST
==================================================

Create at least one test equivalent to our live failure:

- normal car start
- ball at normal distance

Then another:

- same car state
- manually place ball close to the car

The teacher must behave sensibly in BOTH.

If the ball is near the car, it should not simply orbit around it.

==================================================
LIVE RLBot TEST
==================================================

After RocketSim validation, do NOT train.

Deploy the SAME teacher logic into the existing python-example only for a diagnostic live test.

This is allowed because the teacher is not the final ML bot.

The purpose is to verify that the teacher's geometric logic corresponds between:
- RocketSim
- real RLBot

Watch:
- normal kickoff
- ball pushed near car
- ball left/right
- ball behind

We want the teacher to approach the ball sensibly in real Rocket League.

==================================================
DEMONSTRATION DATA DESIGN
==================================================

Do NOT generate a huge dataset yet.

First make a small pilot dataset.

Each record should contain at minimum:

- 92D observation
- teacher action index
- car state if useful for analysis
- ball state if useful for analysis
- target point
- scenario metadata
- episode ID
- timestep

Split by EPISODE, not random rows.

We need:
- training episodes
- validation episodes
- held-out test episodes

Do not leak adjacent frames from the same episode across train/test.

==================================================
AFTER THE TEACHER TEST
==================================================

STOP.

Do NOT build BC yet.

Do NOT train.

Give me a report containing:

1. Exact teacher algorithm.
2. Exact target-point calculation.
3. Exact action-selection logic.
4. Whether it converges to the ball.
5. Whether it avoids circling.
6. Ground-contact rate.
7. Contact rate.
8. Useful-touch rate.
9. Ball movement toward opponent goal.
10. Performance on the 14 controlled scenarios.
11. Performance on randomized states.
12. Live RLBot observations.
13. Failure cases.
14. Whether this teacher is good enough to generate BC demonstrations.
15. What must be fixed before demonstration generation.

==================================================
VERY IMPORTANT
==================================================

Do NOT implement:
- behavior cloning
- PPO fine-tuning
- RelativeDefaultObs
- new neural architecture

until the teacher is independently validated.

The current goal is only:

BUILD A WEAK, SIMPLE, ROBUST TEACHER THAT CAN APPROACH THE BALL AND MAKE BASIC USEFUL TOUCHES TOWARD THE OPPONENT GOAL.

If the teacher fails, fix the teacher.

If the teacher succeeds, stop and report results.

No large training run yet.