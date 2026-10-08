from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

class Robert:
    def __init__(self):
        
        # Modèles utilisés 
        self.vecto=TfidfVectorizer()
        self.model = MultinomialNB()
        self.categ_model = DecisionTreeClassifier()
        
        # Nombre de catégories que Robert fini par comprendre 
        self.nmbrecaté=[]
        
    # Méthode pour entrainer robert (Les phrases, la catégories, et leurs signification)
    
    def train(self,phrases,y,y2):
        # La fonction de base (si tout se passe bien)
        try:
            self.X=self.vecto.fit_transform(phrases)
            self.model.fit(self.X, y)
            
            k=[[i] for i in y]
            
            self.categ_model.fit(k, y2)
            
            self.y=y
            self.y2=y2
        # Si il y a une quelconque erreur noté e0
        except Exception as e0:
            print(f"Erreur dans l'entraînement : {e0}") # Spécifie l'erreur
            print(f"k : {len(k)}, y : {len(y)}, y2 : {len(y2)}, phrases : {len(phrases)}") # Donne la taille des différentes listes (au cas où il y a l'erreur)
            return f"Erreur dans l'entraînement : {e0}" 

    
    
    def convert(self,phrase):
        return self.vecto.transform(phrase)[0]
    
    def textualiser(self,élément):
        return self.categ_model.predict([élément])
    
    def nmbre_cat(self):
        try:
            for i in self.y:
                if i not in self.nmbrecaté:
                    self.nmbrecaté.append(i)
                else:
                    pass
            return self.nmbrecaté
        except Exception as e1:
            print(f"Erreur dans le compte des catégories : {e1}")
            return f"Erreur dans le compte des catégories : {e1}"
        
    def déterminer(self,phrase):
        try:
            print(f"Je sais c'est dans la catégorie {self.categ_model.predict([self.model.predict(self.convert(phrase))])[0]}")
        except Exception as e2:
            print(f"Erreur dans la détermination : {e2}")
            return f"Erreur dans la détermination : {e2}"


phrases = [
    # --- 0 : Salutations & Politesse (20) ---
    "bonjour comment vas tu aujourd'hui",
    "salut ca va bien et toi",
    "hello bonsoir a tous",
    "coucou j'espere que tu vas bien",
    "bonjour bot comment tu te portes",
    "salut comment se passe ta journee",
    "bonsoir j'espere que tout va bien pour toi",
    "hey salut comment allez vous",
    "bonjour ravi de te parler",
    "salut quoi de neuf aujourd'hui",
    "bonjour passe une excellente journee",
    "salut j'espere que tu formes une bonne journee",
    "hello comment ca va ce matin",
    "bonsoir tout le monde",
    "coucou comment tu vas",
    "salut un plaisir de te revoir",
    "bonjour a toi mon cher assistant",
    "hey comment se passe ton week end",
    "salut en forme aujourd'hui",
    "bonjour que puis je faire pour toi",

    # --- 1 : Météo & Climat (20) ---
    "quel temps fait il demain matin",
    "est ce qu'il va pleuvoir cet apres midi",
    "donne moi la meteo de la semaine",
    "fait il beau dehors aujourd'hui",
    "quelle est la temperature actuelle",
    "va t il neiger ce week end",
    "est ce qu'il y aura du soleil demain",
    "quel est le bulletin meteo pour paris",
    "y aura t il du vent ce soir",
    "donne moi les previsions pour demain",
    "est ce qu'il fait chaud dehors",
    "faut il prendre un parapluie aujourd'hui",
    "quelle est la meteo pour ce week end",
    "va t il y avoir un orage ce soir",
    "donne moi le temps pour la ville de lyon",
    "est ce qu'il fait nuit",
    "quel est le taux d'humidite aujourd'hui",
    "y a t il du brouillard ce matin",
    "quelle est la temperature ressentie",
    "est ce que le soleil va se lever bientôt",

    # --- 2 : Musique & Audio (20) ---
    "mets de la musique dynamique",
    "joue une chanson de jazz",
    "lance ma playlist preferee",
    "mets pause sur le morceau",
    "mets le titre suivant s'il te plait",
    "joue le dernier album de cet artiste",
    "lance de la musique douce pour dormir",
    "augmente le volume de la musique",
    "mets une chanson des annees 80",
    "stop la musique tout de suite",
    "baisser le son du lecteur",
    "joue du rock classique",
    "met en pause la piste audio",
    "passe au morceau precedent",
    "mets de la musique de fond",
    "lance un morceau de piano",
    "mets cette chanson en boucle",
    "joue ma liste de lecture pour le sport",
    "coupe le son du haut parleur",
    "mets de la musique relaxante",

    # --- 3 : Rappels, Alarmes & Minuteurs (20) ---
    "mets une alarme pour sept heures",
    "rappelle moi d'acheter du pain ce soir",
    "programme un minuteur de dix minutes",
    "ajoute un rappel pour ma reunion",
    "reveille moi demain a huit heures",
    "programme une alarme a six heures trente",
    "rappelle moi de passer a la pharmacie",
    "mets un minuteur de cuisson pour les pates",
    "ajoute un rendez vous a mon agenda demain",
    "rappelle moi de passer un coup de fil a 14h",
    "mets une alarme pour la priere",
    "programme un réveil pour lundi prochain",
    "mets un minuteur de trente secondes",
    "rappelle moi d'eteindre le four dans vingt minutes",
    "ajoute une note de rappel pour mes devoirs",
    "reveille moi dans deux heures",
    "mets une alarme quotidienne a sept heures",
    "rappelle moi de prendre mes medicaments",
    "ajoute un rappel pour l'anniversaire de paul",
    "programme une alarme pour la fin du cours"
]

y_ = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 20 Salutations
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 20 Météo
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # 20 Musique
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3   # 20 Rappels
]

y_2 = [
    "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation", "salutation",
    "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo", "météo",
    "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique", "musique",
    "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels", "rappels"
]

if __name__=="__main__":
    A=Robert()
    A.train(phrases,y_,y_2)
    A.déterminer(["rock"])

