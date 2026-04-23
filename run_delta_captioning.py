import os
import sys

# Ensure we are in the project root
sys.path.append("/home/ghaffar/nn-gpt")

from ab.gpt.util.AlterNN import AlterCaptionNN

def main():
    alter = AlterCaptionNN()
    # Run the automated improvement loop
    # Passing nn_filter="Blip2Fast" to target our specific model
    alter.alter_delta(nn_filter="Blip2Fast", n=0)

if __name__ == "__main__":
    main()
