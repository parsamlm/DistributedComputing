import argparse
import client_server
import p2p

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="p2p", help="You should choose between 'p2p' and 'client_server'")
args = parser.parse_args()

if args.config == "p2p":
    p2p.run()
elif args.config == "client_server":
    client_server.run()
else:
    raise Exception("Wrong config set.")
