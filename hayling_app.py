import customtkinter as ctk
from PIL import Image
import threading
import speech_recognition as sr
import time
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "C:/Users/hamdi/Desktop/ICube/EMOHayling/checkpoint-363"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

PHRASES = [
    "Lorsqu'elle a appris la mauvaise nouvelle, elle a versé des",
    "Les deux mariés sont partis en voyage de",
    "Il a posté la lettre sans y mettre un",
    "Pour se protéger de la pluie, il a ouvert son",
    "Le bébé pleure pour appeler sa",
    "Le menuisier a cloué un clou avec un",
    "Pour améliorer sa vision, il porte des",
    "Après sa journée de travail, il est rentré à la",
    "Il était tellement bizarre, on aurait dit qu'il venait d'une autre",
    "Il est bon de manger trois fois par",
    "Le chat court après la",
    "Avant d'aller au lit, on éteint la",
    "Il y a beaucoup de livres dans la",
    "On dépose notre argent à la",
    "Pendant le repas, toute la famille est assise autour de la"
]
MOTS_A_INHIBER = [
    "larmes", "noce", "timbre", "parapluie", "maman", "marteau",
    "lunettes", "maison", "planète", "jour", "souris", "lumière",
    "bibliothèque", "banque", "table"
]

def nettoyer_reponse(rep):
    return rep.lower().strip()

def score_auto(stem, response, idx, previous_responses):
    rep_clean = nettoyer_reponse(response)
    mot_inhibe = MOTS_A_INHIBER[idx]
    if rep_clean == mot_inhibe:
        return 3
    if rep_clean in [nettoyer_reponse(r) for r in previous_responses]:
        return 1
    inputs = tokenizer(stem, rep_clean, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
    return pred

class StartScreen(ctk.CTkFrame):
    def __init__(self, master, on_start):
        super().__init__(master, fg_color="#f5f9fc", corner_radius=18, width=470, height=340)
        self.place(relx=0.5, rely=0.5, anchor="center")
        logo_frame = ctk.CTkFrame(self, fg_color="transparent")
        logo_frame.pack(fill="x", pady=(10, 0))
        icube_img = ctk.CTkImage(light_image=Image.open("C:/Users/hamdi/Desktop/ICube/app/laboratoire-icube-logo-png_seeklogo-401018-removebg-preview.png"), size=(55, 55))
        ctk.CTkLabel(logo_frame, image=icube_img, text="", bg_color="transparent").pack(side="left", padx=(18, 0))
        gaia_img = ctk.CTkImage(light_image=Image.open("C:/Users/hamdi/Desktop/ICube/app/Logo_gaia_large-removebg-preview.png"), size=(60, 60))
        ctk.CTkLabel(logo_frame, image=gaia_img, text="", bg_color="transparent").pack(side="right", padx=(0, 18))
        ctk.CTkLabel(self, text="Bienvenue !", font=("Arial", 24, "bold"), text_color="#1d5d99").pack(pady=(16, 7))
        ctk.CTkLabel(self, text="Ce test mesure l'inhibition verbale.\nChoisissez votre mode d'entrée :", font=("Arial", 16), wraplength=390, justify="center").pack(pady=(7, 10))
        self.mode = ctk.StringVar(value="speech")
        radio1 = ctk.CTkRadioButton(self, text="🎤 Microphone", variable=self.mode, value="speech", font=("Arial", 14))
        radio2 = ctk.CTkRadioButton(self, text="⌨️ Clavier", variable=self.mode, value="clavier", font=("Arial", 14))
        radio1.pack(anchor="w", padx=50, pady=5)
        radio2.pack(anchor="w", padx=50, pady=5)
        ctk.CTkButton(self, text="Démarrer le test", command=lambda: on_start(self.mode.get()), fg_color="#1d5d99", font=("Arial", 15, "bold")).pack(pady=(22, 7))

class HaylingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Test d'Inhibition de Hayling (Démo)")
        self.geometry("850x570")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.configure(bg="#f2f6fa")
        self.responses = []
        self.times = []
        self.scores = []
        self.mode = "speech"
        self.listening = False
        self.index = 0
        self.start_screen = StartScreen(self, self.launch_test)
        self.card = None

    def launch_test(self, mode):
        self.mode = mode
        self.start_screen.destroy()
        self.build_ui()
        self.index = 0
        self.responses = []
        self.times = []
        self.scores = []
        self.update_phrase_and_listen()

    def build_ui(self):
        self.card = ctk.CTkFrame(self, fg_color="#f5f9fc", border_width=2, border_color="#1d5d99", corner_radius=18)
        self.card.pack(pady=18, padx=10, fill="both", expand=True)
        logo_frame = ctk.CTkFrame(self.card, fg_color="transparent")
        logo_frame.pack(fill="x", pady=(8, 0))
        icube_img = ctk.CTkImage(light_image=Image.open("C:/Users/hamdi/Desktop/ICube/app/laboratoire-icube-logo-png_seeklogo-401018-removebg-preview.png"), size=(55, 55))
        ctk.CTkLabel(logo_frame, image=icube_img, text="", bg_color="transparent").pack(side="left", padx=(15, 0))
        gaia_img = ctk.CTkImage(light_image=Image.open("C:/Users/hamdi/Desktop/ICube/app/Logo_gaia_large-removebg-preview.png"), size=(60, 60))
        ctk.CTkLabel(logo_frame, image=gaia_img, text="", bg_color="transparent").pack(side="right", padx=(0, 18))
        self.phrase_label = ctk.CTkLabel(self.card, text="", font=("Arial", 22, "bold"), text_color="#1d5d99", wraplength=700, anchor="center", justify="center")
        self.phrase_label.pack(pady=(32, 18), padx=20)
        self.response_entry = ctk.CTkEntry(self.card, width=420, font=("Arial", 18), height=38, corner_radius=11, state="normal")
        self.response_entry.pack(pady=(28, 18))
        if self.mode == "clavier":
            self.response_entry.configure(state="normal")
            self.response_entry.bind('<Return>', lambda e: self.handle_clavier())
            self.bind('<Return>', lambda e: self.handle_clavier())
        else:
            self.response_entry.configure(state="disabled")
        btn_frame = ctk.CTkFrame(self.card, fg_color="transparent")
        btn_frame.pack(pady=(7, 12))
        self.summary_btn = ctk.CTkButton(btn_frame, text="Afficher Résumé", command=self.show_summary, fg_color="#888")
        self.summary_btn.pack(side="left", padx=(0, 10))
        self.export_btn = ctk.CTkButton(btn_frame, text="Exporter CSV", command=self.export_csv, fg_color="#1d5d99")
        self.export_btn.pack(side="left", padx=(10, 0))
        self.footer = ctk.CTkLabel(self, text="Powered by ICube & GAIA – Pour l'évaluation neuropsychologique", font=("Arial", 11), text_color="#1d5d99")
        self.footer.pack(side="bottom", pady=8)

    def update_phrase_and_listen(self):
        if self.index < len(PHRASES):
            self.phrase_label.configure(text=f"{self.index+1}. {PHRASES[self.index]} ...", text_color="#1d5d99")
            self.response_entry.delete(0, 'end')
            self.response_entry.configure(state="normal" if self.mode == "clavier" else "disabled")
            self.start_time = time.time()
            if self.mode == "clavier":
                self.response_entry.focus_set()
            else:
                threading.Thread(target=self.recognize_speech_and_advance).start()
        else:
            self.phrase_label.configure(text="Toutes les phrases sont terminées ! 🎉\nMerci.", text_color="#119c3c")
            self.response_entry.pack_forget()

    def recognize_speech_and_advance(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            while True:
                try:
                    audio = recognizer.listen(source, timeout=180)
                    response = recognizer.recognize_google(audio, language='fr-FR')
                except Exception:
                    response = ""
                if not response:
                    self.afficher_erreur("Aucune parole détectée, réessayez.")
                    continue
                if len(response.split()) != 1:
                    self.afficher_erreur("Veuillez ne dire qu'UN seul mot !")
                    continue
                break
        self.process_and_score(response)

    def handle_clavier(self):
        response = self.response_entry.get().strip()
        if not response:
            self.afficher_erreur("Veuillez saisir une réponse.")
            return
        if len(response.split()) != 1:
            self.afficher_erreur("Veuillez ne saisir qu'un seul mot !")
            return
        self.process_and_score(response)

    def process_and_score(self, response):
        # Cette méthode est appelée EXACTEMENT UNE FOIS par phrase
        elapsed = time.time() - self.start_time if self.start_time else 0
        idx = self.index
        stem = PHRASES[idx]
        previous = self.responses
        score = score_auto(stem, response, idx, previous)
        self.responses.append(response)
        self.times.append(elapsed)
        self.scores.append(score)
        self.index += 1
        self.after(500, self.update_phrase_and_listen)

    def afficher_erreur(self, msg):
        self.phrase_label.configure(text=msg, text_color="#ff3333")
        self.after(1200, lambda: self.phrase_label.configure(
            text=f"{self.index+1}. {PHRASES[self.index]} ...", text_color="#1d5d99"))

    def show_summary(self):
        win = ctk.CTkToplevel(self)
        win.title("Résumé Cotations")
        win.geometry("690x540")
        win.configure(bg="#f5f9fc")
        logo_frame = ctk.CTkFrame(win, fg_color="transparent")
        logo_frame.pack(fill="x", pady=(7,0))
        icube_img = ctk.CTkImage(light_image=Image.open("C:/Users/hamdi/Desktop/ICube/app/laboratoire-icube-logo-png_seeklogo-401018-removebg-preview.png"), size=(38, 38))
        ctk.CTkLabel(logo_frame, image=icube_img, text="", bg_color="transparent").pack(side="left", padx=(8,0))
        gaia_img = ctk.CTkImage(light_image=Image.open("C:/Users/hamdi/Desktop/ICube/app/Logo_gaia_large-removebg-preview.png"), size=(40, 40))
        ctk.CTkLabel(logo_frame, image=gaia_img, text="", bg_color="transparent").pack(side="right", padx=(0,10))
        table = ctk.CTkFrame(win, fg_color="#f5f9fc")
        table.pack(fill="both", expand=True, pady=(13,15), padx=20)
        headers = ["#", "Phrase", "Réponse", "Temps (s)", "Cotation"]
        hframe = ctk.CTkFrame(table, fg_color="#f2f6fa")
        hframe.pack(fill="x")
        for i, h in enumerate(headers):
            ctk.CTkLabel(hframe, text=h, font=("Arial", 14, "bold"), width=90, text_color="#1d5d99").grid(row=0, column=i, padx=4, pady=4)
        for i, (phrase, r, t, s) in enumerate(zip(PHRASES, self.responses, self.times, self.scores)):
            if s == 0:
                bg = "#27bf56"
                fg = "#fff"
            elif s == 1:
                bg = "#ffa500"
                fg = "#222"
            elif s == 3:
                bg = "#d6404f"
                fg = "#fff"
            else:
                bg = "#aaa"
                fg = "#000"
            rowf = ctk.CTkFrame(table, fg_color="#f5f9fc")
            rowf.pack(fill="x", pady=0)
            ctk.CTkLabel(rowf, text=str(i+1), font=("Arial", 13), width=24, text_color="#1d5d99").grid(row=0,column=0)
            ctk.CTkLabel(rowf, text=phrase, font=("Arial", 12), width=235, anchor="w", text_color="#1d5d99").grid(row=0,column=1)
            ctk.CTkLabel(rowf, text=r, font=("Arial", 13), width=98, anchor="center").grid(row=0,column=2)
            ctk.CTkLabel(rowf, text=f"{t:.2f}", font=("Arial", 13), width=52, anchor="center").grid(row=0,column=3)
            ctk.CTkLabel(rowf, text=str(s), font=("Arial", 13, "bold"), width=38, fg_color=bg, text_color=fg, corner_radius=12).grid(row=0,column=4, padx=4)

    def export_csv(self):
        filename = "resultats_hayling_demo.csv"
        with open(filename, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Phrase", "Réponse", "Temps", "Cotation"])
            for i, (r, t, s) in enumerate(zip(self.responses, self.times, self.scores)):
                writer.writerow([PHRASES[i], r, f"{t:.2f}", s])


if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    app = HaylingApp()
    app.mainloop()
