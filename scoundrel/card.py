class Card:
    def __init__(self, suit, value, type):
        self.suit = suit
        self.value = value
        if suit == "hearts":
            self.type = "potion"
        elif suit == "diamonds":
            self.type = "weapon"
        elif suit == "spades" or suit == "clubs":
            self.type = "monster"
        else:
            raise ValueError(f"Invalid suit: {suit}")
        
    def __str__(self):
        return f"{self.value} of {self.suit}"

    def __repr__(self):
        return f"Card(suit={self.suit}, value={self.value}, type={self.type})"
    
    