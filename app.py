import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from auth_token import AuthToken

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app

app = tk.Tk()
app.geometry("532x622")
app.title("Stable Diffusion")

ctk.set_appearance_mode("dark")
prompt = ctk.CTkEntry(height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white", master=app)
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(height=512, width=512, master=app)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid,
                                               # revision="fp16",
                                               # torch_dtype=torch.float16,
                                               use_auth_token=AuthToken)
pipe.to(device)
def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    img = ImageTk.PhotoImage(image)
    img.save("generated_image.png")
    lmain.configure(image=img)


trigger = ctk.CTkButton(height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", master=app,
                        command=generate)
trigger.configure(text="Generate")
trigger.place(x=206,y=60)


app.mainloop()