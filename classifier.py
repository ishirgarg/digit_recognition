# Used by app.py for machine learning functions
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

NUM_CLASSES = 16

def int_to_class(predicted_class):
    dict = {
        0 : '0',
        1 : '1',
        2 : '2',
        3 : '3',
        4 : '4',
        5 : '5',
        6 : '6',
        7 : '7',
        8 : '8',
        9 : '9',
        10 : '+',
        11 : '-',
        12 : '*',
        13 : '/',
        14 : '[',
        15 : ']'
    }
    return dict[predicted_class]

# Function for creating model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((28, 28, 1)),
        tf.keras.layers.Rescaling(1./255, input_shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(rate=0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(rate=0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES),
        tf.keras.layers.Softmax()
    ])

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.00005),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    
    return model

# Load the model parameters
def load_model(file_input_path):
    # Load model
    model = create_model()
    model.load_weights(file_input_path)

    return model

# Check if an image is valid (not a few pixel anomaly)
def valid_image_check(image):
    VALID_IMAGE_THRESHOLD = 12 # Threshold is pretty arbitrary; min number of pixels that must be colored

    total = 0
    for i in range(28):
        for j in range(28):
            if (image[i][j] > 0):
                total += 1
    return total > VALID_IMAGE_THRESHOLD

# Process the digits into bounding boxes
def process_digits(img_file_path, display_bounding_boxes=False):
    # Predictions on images with multiple digits
    image = cv2.imread(img_file_path)

    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preprocessed_digits = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        # Creating bounding box
        if display_bounding_boxes:
            cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
            
        
        # Cropping out the digit from the image
        digit = thresh[y:y+h, x:x+w]
        
        # Resize image, but don't disproportionately scale it
        resized_width = int(24 * w / max(w, h))
        resized_height = int(24 * h / max(w, h))
        if resized_width % 2 == 1: resized_width += 1 # Must scale to even number so padding is same on both sides
        if resized_height % 2 == 1: resized_height += 1 # Must scale to even number so padding is same on both sides

        resized_digit = cv2.resize(digit, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        _, resized_digit = cv2.threshold(resized_digit, 25, 255, cv2.THRESH_BINARY)

        remaining_pad_width = (28 - resized_width) // 2
        remaining_pad_height = (28 - resized_height) // 2
        padded_digit = np.pad(resized_digit, ((remaining_pad_height, remaining_pad_height),(remaining_pad_width, remaining_pad_width)), "constant", constant_values=0)

        if (valid_image_check(padded_digit)):
            preprocessed_digits.append([padded_digit, x + w / 2, y + h / 2])

    if display_bounding_boxes:
        plt.imshow(image, cmap='gray')
        plt.show()

    return preprocessed_digits

 # preprocessed_digits is an array where each element is a 3-element array:

# [image of char, x-coord of bounding box center, y-coord of bounding box center]
def form_expression(model, preprocessed_digits):
    sorted_digits = sorted(preprocessed_digits, key=lambda x : x[1])

    formatted_expression = ""
    for digit in sorted_digits:
        formatted_expression += int_to_class(np.argmax(model.predict(digit[0].reshape(1, 28, 28))))

    return formatted_expression

#Expression evaluation
# Custom exception for invalid expressions
class MalformedExpression(Exception):
    def __init__(self, message=''):
        super().__init__(message)

# Check if expr[i] is a negative sign
def is_negative(expr, i):
    if not (expr[i] == '-'):
        return False
    
    if (i < len(expr) - 1 and (expr[i + 1].isdigit() or expr[i + 1] == '[')):
        if (i > 0) and not expr[i - 1].isdigit() and not expr[i - 1] == ']':
            return True
        if (i == 0):
            return True
        
    return False

# Check if expr[i] is a negaative sign OR a digit
def is_digit(expr, i):
    return is_negative(expr, i) or expr[i].isdigit()

# Check if c is an operator character
def is_operator(c):
    return c == '*' or c == '/' or c == '+' or c == '-'
        
# Check if expr is a valid mathematical expression
def check_expression(expr):
    # Check that parentheses align, and there is no ()
    def check_parentheses(expr):
        paren_count = 0
        for i in range(len(expr)):
            c = expr[i]
            if (c == '['):
                paren_count += 1
            elif (c == ']'):
                paren_count -= 1
            if (paren_count < 0):
                return False
            if (c == ']') and expr[i - 1] == '[':
                return False
            
        return paren_count == 0
    
    # Make sure that '[' is always followed by a digit, another '[', or a negative sign
    def check_number_after_parentheses(expr):
        for i in range(len(expr)):
            if (expr[i] == '[') and not ((expr[i + 1]).isdigit() or expr[i + 1] == '[' or is_negative(expr, i + 1)):
                return False
        return True
    
    # Make sure that the operators are correctly formed
    def check_operators(expr):
        for i in range(len(expr)):
            # Ignore negative signs; they are part of digits
            if (is_negative(expr, i)):
                continue

            # If an operator is found
            if is_operator(expr[i]):
                # Operators cannot be at start or end
                if (i == 0 or i == len(expr) - 1):
                    return False
                # Operators must have either a ) or a digit before them
                if not (expr[i - 1] == ']' or expr[i - 1].isdigit()):
                    return False
                # Operators must have either a ( or a digit/negative sign after them
                if not (expr[i + 1] == '[' or is_digit(expr, i + 1)):
                    return False
        return True
     
    return len(expr) > 0 and check_parentheses(expr) and check_number_after_parentheses(expr) and check_operators(expr)

# Convert an expression to an array where ints are combined into one entry
def expression_to_array(expr):
    # Make sure expression is valid
    if not check_expression(expr):
        raise MalformedExpression("Malformed expression: " + expr)

    expr_arr = []
    index = 0
    while index < len(expr):
        # Case 1: we are at a number
        if (is_digit(expr, index)):
            # Get the number
            num = ""
            while (index < len(expr)) and is_digit(expr, index):
                num += expr[index]
                index += 1
            # Add number to array
            if (num == '-'):
                expr_arr.append(-1)
            else:
                expr_arr.append(int(num))
        # Case 2: parentheses or operator
        else:
            expr_arr.append(expr[index])
            index += 1

    return expr_arr

# Compute the value of an expression; returns value as an array
def solve_expression_arr(expr):
    def concat_arrays(arr_list):
        new_arr = []
        for arr in arr_list:
            for element in arr:
                if (element != ''):
                    new_arr.append(element)
        return new_arr
    
    # Return expression if expression is just a number
    if (len(expr) == 1):
        return expr
    
    # Find first pair of parentheses
    firstParenIndex = -1
    lastParenIndex = -1
    for i in range(len(expr)):
        if (expr[i] == '['):
            firstParenIndex = i
            paren_count = 0
            for i in range(firstParenIndex, len(expr)):
                if (expr[i] == '['):
                    paren_count += 1
                elif (expr[i] == ']'):
                    paren_count -= 1
                if (paren_count == 0):
                    lastParenIndex = i
                    break
            break

    # If parentheses are found
    if (firstParenIndex != -1):
        operator1 = '' # May need to insert * operator in a case like 3(5) --> 3 * 5
        operator2 = '' # May need to insert * operator in a case like 3(5) --> 3 * 5
        if (firstParenIndex > 0 and type(expr[firstParenIndex - 1]) != str):
            operator1 = '*'
        if (lastParenIndex < len(expr) - 1) and type(expr[lastParenIndex + 1]) != str:
            operator2 = '*'
        
        sub_expression = solve_expression_arr(expr[firstParenIndex + 1:lastParenIndex])
        # Case 1: we end up with a double negative such as -[-3] --> 3
        return solve_expression_arr(concat_arrays((expr[0:firstParenIndex], [operator1], sub_expression, [operator2], expr[lastParenIndex + 1:])))
    
    # Evaluate all * and / operations
    for i in range(len(expr)):
        if (expr[i] == '*'):
            return solve_expression_arr(concat_arrays((expr[0:i - 1], [expr[i - 1] * expr[i + 1]], expr[i + 2:])))
        
        if (expr[i] == '/'):
            if (expr[i + 1] == 0):
                raise ZeroDivisionError("Expression attempts to divide by 0")

            return solve_expression_arr(concat_arrays((expr[0:i - 1], [expr[i - 1] / expr[i + 1]], expr[i + 2:])))
        
    # Evaluate all '+' and '-' expressions
    for i in range(len(expr)):
        if (expr[i] == '+'):
            return solve_expression_arr(concat_arrays((expr[0:i - 1], [expr[i - 1] + expr[i + 1]], expr[i + 2:])))
        
        if (expr[i] == '-'):
            return solve_expression_arr(concat_arrays((expr[0:i - 1], [expr[i - 1] - expr[i + 1]], expr[i + 2:])))

# Compute the value of an expression; returns value as an int/double
def solve_expression(expr):
    expr_arr = expression_to_array(expr)
    return solve_expression_arr(expr_arr)[0]

# Evaluate the expression in the image
def evaluate_image_expression(model, img_file_path):
    try:
        expr = form_expression(model, process_digits(img_file_path))
        return [expr, solve_expression(expr)]
    except ZeroDivisionError:
        return "Divide by 0 error: " + expr
    except MalformedExpression:
        return "Invalid expression: " + expr
