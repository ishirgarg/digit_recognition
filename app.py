import sys
import pygame
import os
from classifier import load_model, evaluate_image_expression

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

if __name__ == '__main__':
    pygame.init()
    fps = 500
    fps_clock = pygame.time.Clock()
    width, height = 800, 640
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    font = pygame.font.SysFont('Arial', 20)

    # Load model
    model = load_model(resource_path("expression_classifier_model.ckpt"))

    # Text to display for expression
    expression_text = ""

    # Brush
    draw_color = [0, 0, 0]
    brush_size = 5
    # Canvas size
    canvas_size = [1500, 600]

    def clear_canvas():
        canvas.fill((255, 255, 255))

    def save_canvas(file_path):
        pygame.image.save(canvas, file_path)

    def evaluate_expression():
        global expression_text

        file_path = resource_path("canvas.png")
        save_canvas(file_path)

        eval_result = evaluate_image_expression(model, file_path)
        if (type(eval_result) == str):
            expression_text = eval_result
        else:
            expression_text = eval_result[0] + ' = ' + str(round(eval_result[1], 4)) 

    class Button():
        def __init__(self, x, y, width, height, button_text, on_click_function=None, one_press=False):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.on_click_function = on_click_function
            self.one_press = one_press

            self.fill_colors = {
                'normal': '#ffffff',
                'hover': '#666666',
                'pressed': '#333333',
            }

            self.button_surface = pygame.Surface((self.width, self.height))
            self.button_rect = pygame.Rect(self.x, self.y, self.width, self.height)

            self.button_surf = font.render(button_text, True, (20, 20, 20))
            self.already_pressed = False
        
        def process(self):
            mouse_pos = pygame.mouse.get_pos()

            self.button_surface.fill(self.fill_colors['normal'])
            if self.button_rect.collidepoint(mouse_pos):
                self.button_surface.fill(self.fill_colors['hover'])

                if pygame.mouse.get_pressed(num_buttons=3)[0]:
                    self.button_surface.fill(self.fill_colors['pressed'])

                    if self.one_press:
                        self.on_click_function()

                    elif not self.already_pressed:
                        self.on_click_function()
                        self.already_pressed = True

                else:
                    self.already_pressed = False

            self.button_surface.blit(self.button_surf, [
                self.button_rect.width/2 - self.button_surf.get_rect().width/2,
                self.button_rect.height/2 - self.button_surf.get_rect().height/2
            ])
            screen.blit(self.button_surface, self.button_rect)

    # White background canvas
    canvas = pygame.Surface(canvas_size)
    clear_canvas()

    while True:
        screen.fill((30, 30, 30))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Reset canvas button
        canvas_button = Button(30, 50, 100, 50, "Evaluate", evaluate_expression)
        canvas_button.process()
        # Evaluate button
        evaluate_button = Button(150, 50, 100, 50, "Reset", clear_canvas)
        evaluate_button.process()

        # Draw the Canvas at the center of the screen
        x, y = screen.get_size()
        screen.blit(canvas, [x/2 - canvas_size[0]/2, y/2 - canvas_size[1]/2])

        # Text to display expression result
        font = pygame.font.SysFont('Arial', 20)
        text = font.render(expression_text, True, (0, 0, 0), (255, 255, 255))

        CANVAS_BOX_MARGIN = 10 # Margin between bottom of canvas and top edge of textbox
        BOX_HEIGHT = 50 # Height of text box
        BOX_LEFT_MARGIN = 25 # Margin between left edge of canvas and left edge of textbox

        # Display expression text
        text_rect = text.get_rect()
        text_rect.center = (x / 2, y / 2 + canvas_size[1] / 2 + CANVAS_BOX_MARGIN + BOX_HEIGHT / 2)
        background_rect = pygame.Rect(x / 2 - canvas_size[0] / 2 + BOX_LEFT_MARGIN, 
                            y / 2 + canvas_size[1] / 2 + CANVAS_BOX_MARGIN, 
                            canvas_size[0] - 2 * BOX_LEFT_MARGIN, 
                            BOX_HEIGHT, draw_color=(255, 255, 255))
    
        pygame.draw.rect(screen, (255, 255, 255), background_rect)
        screen.blit(text, text_rect)

        # Drawing with the mouse
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()

            # Calculate Position on the Canvas
            dx = mx - x/2 + canvas_size[0]/2
            dy = my - y/2 + canvas_size[1]/2

            pygame.draw.circle(
                canvas,
                draw_color,
                [dx, dy],
                brush_size,
            )

        pygame.display.flip()
        fps_clock.tick(fps)