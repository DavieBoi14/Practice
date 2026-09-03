import math 
import random

print("I've picked a random number 0-100")


answer = random.randint(0, 100) 
num_guesses = 0

guess = int(input("Take a guess... "))

while (guess != answer):
    if (guess > answer):
        guess = int(input("Too high, try again... "))
    else:
        guess = int(input("Too low, try again... "))
    num_guesses += 1

print("CORRECT!")
print(f"You guessed the number {guess} corectly in {num_guesses} guesses!")