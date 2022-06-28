# Run with Docker

‘sudo docker build -t rl-breakout-image .‘
‘sudo docker run --rm -v ${PWD}/games/:/app/games rl-breakout-image play_game.py --agent 'ddqn' --gif_name 'ddqn-agent.gif'‘
