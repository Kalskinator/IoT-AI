import random
# A.0.1: Cinnamon buns (formatting output) Mandatory - 1pt

# A bakery sells fresh Swedish lovely Kanelbulle for 35 SEK each.
# Day-old Kanelbulle is discounted by 60 percent. 
# A customer enters to buy only old kanelbulle from the store. 
# Write a program that begins by reading the number of day-old Kanelbulle being purchased from the user. 
# Then your program should display the regular price for them, the discount because they are day-old, and the total price. 
# All of the values should be displayed using two decimal places, 
# and the decimal points in all of the numbers should be aligned when reasonable values are entered by the user.

def cinnamon_buns(num_buns):
    price = 35.0
    discount = 0.6

    regular_price = num_buns * price
    discount_price = regular_price * discount
    total_price = regular_price - discount_price

    print("\nPrice Breakdown:")
    print(f"Regular price:   {regular_price:.2f} SEK")
    print(f"Discount:        {discount_price:.2f} SEK")
    print(f"Total price:     {total_price:.2f} SEK")

# A.0.2: Dog Years! (conditionals) Mandatory - 1pt

# It is commonly said that one human year is equivalent to 7 dog years. 
# However, this simple conversion fails to recognize that dogs reach adulthood in approximately two years. 
# As a result, some people believe that it is better to count each of the first two human years as 10.5 dog years, 
# and then count each additional human year as 4 dog years.
# Write a program that implements the conversion from human years to dog years described in the previous paragraph. 
# Ensure that your program works correctly for conversions of less than two human years and for conversions of two or more human years. 
# Your program should display an appropriate error message if the user enters a negative number.

def dog_years(years):
    if years < 0:
        raise ValueError("Input must be a non-negative number.")
    
    if years > 2:
        dog_years = 10.5 + ((years - 2) * 4)
    else:
        dog_years = 5.25 * years

    print(f"Year conversion from {years:.1f} is:  {dog_years:.2f}")

# A.0.4. Is it a prime number? (function) Mandatory - 1pt

# A prime number is an integer greater than 1 that is only divisible by one and itself. 
# Write a function that determines whether or not its parameter is prime, returning True if it is, and False otherwise. 
# Write a main program that reads an integer from the user and displays a message indicating whether or not it is prime.

def prime(num):
    if num <= 1:
        return False
    
    for i in range(2, (num//2) + 1):
        if (num % i) == 0:
            return False
    else:
        return True
    
# A.0.5. What is the next prime? (loop) Mandatory - 1pts

# In this exercise, you will create a function named "nextPrime" that finds and returns  
# the first prime number larger than some integer, n. The value of n will be passed to the function as its only parameter. 
# Include a main program that reads an integer from the user and displays the first prime number larger than the entered value. 
# Import and use your solution to Exercise A.03 while completing this exercise. 

def next_prime(num):
    tmp = num + 1

    while not prime(tmp):
        tmp += 1
    return tmp


# A.0.3: Coin Flip - 3 same flips (loop) Mandatory - 1pts

# Create a program that uses Python’s random number generator to simulate flipping a coin several times. 
# The simulated coin should be fair, meaning that the probability of heads is equal to the probability of tails.
# Each time the program is run, your program should flip simulated coins until either 3 consecutive heads or 3 consecutive tails occur.
# Display an H each time the outcome is heads, and a T each time the outcome is tails, with all of the outcomes shown on the same line. 
# Then display the number of flips needed to reach 3 consecutive flips with the same outcome.  

def coin_flip():
    consecutive = 0
    last_coin = None
    flips = []
    total_flips = 0

    while consecutive < 3: 
        coin = random.choice(["H", "T"]) 
        flips.append(coin)
        total_flips += 1

        if coin == last_coin:
            consecutive += 1
        else:
            consecutive = 1  

        last_coin = coin
    
    print(" ".join(flips) + f" ({total_flips} flips)" )

# A.0.6. Shuffle the cards! (function) Mandatory - 2pt

# A standard deck of playing cards contains 52 cards. 
# Each card has one of four suits along with a value. 
# The suits are normally spades, hearts, diamonds, and clubs while the values are 2 through 10, Jack, Queen, King and Ace. 
# Each playing card can be represented using two characters. 
# The first character is the value of the card, with the values 2 through 9 being represented directly. 
# The characters “T”, “J”, “Q”, “K” and “A” are used to represent the values 10, Jack, Queen, King and Ace respectively. 
# The second character is used to represent the suit of the card. 
# It is normally a lowercase letter: “s” for spades, “h” for hearts, “d” for diamonds and “c” for clubs. 
# The following table provides several examples of cards and their two-character representations. 

# Begin by writing a function named createDeck.
# It will use loops to create a complete deck of cards by adding the suit to the number and 
# storing the two-character abbreviations for all 52 cards in a list. 
# Return the list of cards as the function’s only result. 
# Your function will not take any parameters. 

# Write a second function named shuffle that randomizes the order of the cards in a list. 
# The input of the function should be the list contacting the deck of cards (52 items) that you have created in the previous function. 
# You can use the "random" library shuffle function. 
# Alternatively, you can write your own shuffling function:  
# One technique that can be used to shuffle the cards is to visit each element in the list and swap it with another random element in the list. 
# You must write your own loop for shuffling the cards. 
# The output of the function should be 2 lists: the deck of cards before shuffling and the deck of cards after shuffling.

def createDeck():
    deck = []

    suits = ['s','h','d','c']
    values = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']

    for suit in suits:
        for value in values:
            card = value + suit
            deck.append(card)

    return deck


# Write a second function named shuffle that randomizes the order of the cards in a list. 
def shuffle(deck):
    return deck, random.sample(deck, len(deck))



# A.0.7: Deal the cards (function and class) Mandatory - 3pts
# In many card games, each player is dealt a specific number of cards after the deck has been shuffled. 
# Write a function, deal, which takes the number of hands, the number of cards per hand, and a deck of cards as its three parameters. 
# Your function should return a list containing all of the hands that were dealt. 
# Each hand will be represented as a list of cards. 
# When dealing with the hands, your function should modify the deck of cards passed to it as a parameter, removing each card from the deck as it is added to a player’s hand. 
# When cards are dealt, it is customary to give each player a card before any player receives an additional card.
# Your function should follow this custom when constructing the hands for the players. 
# When done with writing the above function, use your solution to A.0.6 
# to help you construct a main class that creates, shuffles, and deals a deck of cards. 
# Your class structure should be like below:


class cards():

    def __init__(self):
        self.cards = []

    def create (self):
        self.cards = createDeck()

    def shuffle (self):
       _, self.cards = shuffle(self.cards)

    def deal (self,hands,card_num):
        if hands < 0 or card_num < 0:
            raise ValueError("Number of hands and number of cards can't be negative")

        hand_list = [] 
        for _ in range(hands):
            hand_list.append([])

        for _ in range(card_num):  
            for hand in hand_list: 
                if self.cards:  
                    card = self.cards.pop(0) 
                    hand.append(card)
        return hand_list 


if __name__ == "__main__":
    
    # cinnamon_buns(int(input("Enter the number of day-old Kanelbulle being purchased: ")))

    # dog_years(float(input("Enter the number of dog-years you would like to convert to human years: ")))

    # print(prime(int(input("Enter a number you would like to check if its prime: "))))

    # print(next_prime(int(input("Enter an integer you would like a larger prime number of: "))))

    # coin_flip()

    # print(shuffle(createDeck()))

    # initialte your programs with this functions
    card_01 = cards()
    card_01.create()
    card_01.shuffle()
    print(card_01.deal(10,10)) #Change X and Y 

