# Swarmkeeper: design notes

A game built on the Substrata ideas: creatures that learn within a lifetime and evolve across generations. Prototype: `game/swarmkeeper.html` (single file, runs in a browser).

## Direction so far

- **Top-down**, not side-on: the player needs to see many creatures clearly at once.
- **Pet-sized creatures**, roughly a hamster to a human, so they can be cute and personable, in good or bad ways.
- **Pixel art**, inspired by the look and feel of the "Destroy Any Website" stickman game, with a different mechanic.
- **Pikmin-like structure**: you lead and deploy a swarm you have bred. Action happens by day; breeding and evolution happen at night.
- **Feeding is the central verb**: a treat is both a reward that teaches (learning within a life) and food that decides who breeds (evolution).

## Playtest 1 (2026-10-06): feeding as training

- It had promise and was fun even with a touchpad. Nothing felt tedious.
- The fun was **tracking which critters actually responded** and rewarding the right ones before losing track. In learning terms this is credit assignment, done by the player in real time. Working idea: *the player is the reward signal*, and attention is the skill.
- Mistakes teach the wrong thing (a misplaced treat rewards a thief), which produces bad habits without anyone designing them.

## Prototype 2: giving training a payoff

- **Fetching**: critters can carry berries to a larder beside the burrow. The larder feeds everyone at night, so fetchers grow the colony. Fetching starts rare and is taught by rewarding critters that arrive with a berry.
- **Raiders**: a beetle comes to steal from the larder, more often as days pass. Three or more critters together drive it off; a lone critter just gets nipped. The whistle becomes a way to rally the swarm.
- **Readable tells**: ears perk when a critter hears the whistle; responders get a brief glow; hand-feeding a critter beside you is the most precise reward.

## Later: rule-breakers (option D)

Give rule-breaking a specific flavor rather than "eats without working." Treating non-working eaters as the villains carries negative social implications when mapped onto people. The rule-breaker should commit a particular act that harms others.

Candidates to choose from later:
- **Pantry pilferers**: sneak berries out of the larder at night for themselves.
- **Treat snatchers**: intercept treats tossed to other critters, stealing credit as well as food.
- **Whistle mimics**: imitate the keeper's call to lure others away from a task or into a raider's path.

## Open questions

- Scale (same as Pikmin, or a giant among microbes) is settled at pet-sized for now.
- Single player, rooms with friends, or swarm against swarm.
- Whether this joins the Stratum game or stands alone.
