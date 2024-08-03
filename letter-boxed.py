# Evan Gaus - Letter Boxed Solver Project

# ===== ===== ===== ===== ===== Import statements ===== ===== ===== ===== =====

import autograd.numpy as np
from collections import Counter
import nltk
from nltk.corpus import webtext
from nltk.corpus import gutenberg
from nltk.corpus import brown

# Download them to make sure they're up to date
# nltk.download('webtext')
# nltk.download('gutenberg')
# nltk.download('brown')


# ===== ===== ===== ===== ===== Functions ===== ===== ===== ===== =====

def H_word_list_setup():
    # Get the words
    webtext_words = webtext.words()
    gutenberg_words = gutenberg.words()
    brown_words = brown.words()

    # Make them all uppercase
    webtext_words = [word.upper() for word in webtext_words]
    gutenberg_words = [word.upper() for word in gutenberg_words]
    brown_words = [word.upper() for word in brown_words]

    # Combine them
    all_words_dupl = webtext_words + gutenberg_words + brown_words

    # Remove all words that have symbols in them
    all_words_dupl = [word for word in all_words_dupl if word.isalpha()]

    # Remove duplicates
    all_words = list(set(all_words_dupl))

    # Create counter
    word_freq_counter = Counter(all_words_dupl)

    return all_words, word_freq_counter


def H_get_words_based_on_letters(valid_letters, word_list=None):
    # Check if a word list was provided
    if word_list is None:
        # If not, use the our words
        word_list = ALL_WORDS

    # Get rid of words that are only 1 letter long
    word_list = [word for word in word_list if len(word) > 1]

    # Initialize a list to store all valid words
    valid_words = []

    # Make valid letters uppercase
    valid_letters = valid_letters.upper()

    # For all possible words
    for word in word_list:

        # Make word uppercase
        word = word.upper()

        is_valid = True

        for letter in word:
            if letter not in valid_letters:
                is_valid = False
                break

        if is_valid:
            valid_words.append(word)

    # Return the valid words list
    return valid_words


def H_refine_based_on_sides(word_list, side1, side2, side3, side4):
    
    refined_word_list = []

    # For each word
    for word in word_list:
        # Get the string that corresponds to the sides
        temp_string = H_get_temp_string(word, side1, side2, side3, side4)
        
        is_valid = True

        # If there are any repeating numbers in a row, don't add the word
        for i in range(len(temp_string) - 1):
            if temp_string[i] == temp_string[i+1]:
                is_valid = False
                break

        if is_valid:
            refined_word_list.append(word)

    # Return the refined word list
    return refined_word_list


def H_get_temp_string(word, side1, side2, side3, side4):
    temp_string = ''
    for letter in word:
        if letter in side1:
            temp_string += '1'
        elif letter in side2:
            temp_string += '2'
        elif letter in side3:
            temp_string += '3'
        elif letter in side4:
            temp_string += '4'
    return temp_string


def H_get_all_valid_words(side1, side2, side3, side4):
    
    # Capitalize everything jesus
    side1 = side1.upper()
    side2 = side2.upper()
    side3 = side3.upper()
    side4 = side4.upper()

    # Call the letters helper
    valid_letters = side1 + side2 + side3 + side4
    letters_words = H_get_words_based_on_letters(valid_letters)

    # Refine it based on the sides
    refined_words = H_refine_based_on_sides(letters_words, side1, side2, side3, side4)

    return refined_words


def H_get_words_that_start_with_letter(words, letter):
    return [word for word in words if word[0] == letter]


def H_is_attempt_correct(attempt, side1, side2, side3, side4):
    # Make sure the words in attempt use all of the letters in side1 + side2 + side3 + side4
    attempt_letters = ''.join(attempt)
    # print(attempt_letters)
    all_letters = side1 + side2 + side3 + side4
    # print(all_letters)

    for letter in all_letters:
        if letter not in attempt_letters:
            return False
        
    # Else return true
    return True


# Recursive function
def H_recursive_attempt(attempt, all_words, side1, side2, side3, side4):
     
    # Check if the attempt has all the letters
    if H_is_attempt_correct(attempt, side1, side2, side3, side4):
        return attempt, True
    
    # If the attempt is 4 words already, return it
    if len(attempt) >= MAX_ATTEMPT_LENGTH:
        return attempt, False
    
    # If attempt is at least 1 word
    if len(attempt) >= 1:
        # Get the last letter of the last word
        last_letter = attempt[-1][-1]

        # Get all the words that start with that letter
        words_that_start_with = H_get_words_that_start_with_letter(all_words, last_letter)
    else:
        words_that_start_with = all_words

    # For each possible next word
    for word in words_that_start_with:
        # If the word is already in the attempt, skip it
        if word in attempt:
            continue

        # Add it to the attempt
        new_attempt = attempt + [word]

        # Recurse
        new_attempt, is_correct = H_recursive_attempt(new_attempt, all_words, side1, side2, side3, side4)

        # If the attempt is correct, add it to the list of correct attempts
        if is_correct:
            # This is what happens when we have a correct attempt
            # print(f"Correct attempt: {new_attempt}")
            CORRECT_ATTEMPTS.append(new_attempt)
            print('.', end='', flush=True)

    # Once we try all the words that start with the last letter, return the correct attempts
    return attempt, False


# Starter function
def H_start_recursion(all_words, side1, side2, side3, side4):

    # Start the recursion
    _ = H_recursive_attempt([], all_words, side1, side2, side3, side4)

    # Return the correct attempts
    return CORRECT_ATTEMPTS


# Run function
def run(side1, side2, side3, side4):

    # Reset the correct attempts
    CORRECT_ATTEMPTS = []

    # Get the valid words
    list_of_valid_words = H_get_all_valid_words(side1, side2, side3, side4)

    # Call the starter function
    ret_correct_attempts = H_start_recursion(list_of_valid_words, side1, side2, side3, side4)

    # Give each correct attempt a commonality score (using the word_freq_counter defined before)
    commonality_scores = [sum(WORD_FREQ_COUNTER[word] for word in attempt) for attempt in ret_correct_attempts]

    # Sort the lists
    sorted_indicies = np.argsort(commonality_scores)#[::-1]
    ret_correct_attempts = [ret_correct_attempts[i] for i in sorted_indicies]
    commonality_scores = [commonality_scores[i] for i in sorted_indicies]

    # Print some stuff
    print("\n\nCorrect attempts:\n")
    # Enumerate through the correct attempts
    for i, attempt in enumerate(ret_correct_attempts):
        print(f"{i+1}: ({commonality_scores[i]}) {attempt}")
    print(f"\nNumber of correct attempts: {len(ret_correct_attempts):,}\n")


# ===== ===== ===== ===== ===== Main ===== ===== ===== ===== =====

# IMPORTANT VARIABLES ----- ----- -----

MAX_ATTEMPT_LENGTH = 0
CORRECT_ATTEMPTS = []
ALL_WORDS = []
WORD_FREQ_COUNTER = None

# ACTUAL MAIN ----- ----- -----

# Set up the word list
print("Setting up the word list...")
ALL_WORDS, WORD_FREQ_COUNTER = H_word_list_setup()

# Ask if we want to remove rare words
rare_input = input("Do you want to remove rare words? (y/n) ")
rare_input = rare_input.lower()
if rare_input == 'y' or rare_input == 'yes':
    # Remove words that are only used once
    print("Removing rare words...")
    ALL_WORDS = [word for word in ALL_WORDS if WORD_FREQ_COUNTER[word] > 1]

# Ask how long the attempts should be
MAX_ATTEMPT_LENGTH = int(input("How long should the attempts be? "))
# QQQ Maybe make a way to say you want to go until you find a solution (so try 2, try 3, try 4)

# Prompt for the sides
side1 = input("Enter the first side: ")
side2 = input("Enter the second side: ")
side3 = input("Enter the third side: ")
side4 = input("Enter the fourth side: ")

# Send them to upper
side1 = side1.upper()
side2 = side2.upper()
side3 = side3.upper()
side4 = side4.upper()

# Print confirmation
print(f"\n\nRunning with Sides: {side1}, {side2}, {side3}, {side4}")

# Run the function
run(side1, side2, side3, side4)
