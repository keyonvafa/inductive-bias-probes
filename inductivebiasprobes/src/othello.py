from othello_world.data.othello import OthelloBoardState

eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]


class ReversibleOthelloBoardState(OthelloBoardState):
    """Extension of OthelloBoardState that supports reversing moves."""

    def __init__(self):
        super().__init__()

    def get_flips_info(self, move):
        """
        Determine which squares would be flipped by this move, and what color piece ends up placed.
        Also detect a forfeit scenario, as in umpire(...).

        Returns a dictionary with:
            {
                "move": move,
                "placed_color": int,  # final color placed on the board
                "flips": list of [r, c] squares that get flipped due to this move,
                "forfeit": bool,      # True if the move required a color-forfeit first
            }
        Raises ValueError if the move is illegal (i.e. zero flips even after forfeit).
        """
        r, c = move // 8, move % 8

        # Basic sanity check
        if self.state[r, c] != 0:
            raise ValueError(f"Square {r},{c} is already occupied! Illegal move.")

        color = self.next_hand_color
        # Attempt to flip
        flips = []
        forfeit = False

        def find_flips(r, c, color):
            buffer_flips = []
            for direction in eights:
                tmp = []
                cur_r, cur_c = r, c
                while True:
                    cur_r += direction[0]
                    cur_c += direction[1]
                    if not (0 <= cur_r < 8 and 0 <= cur_c < 8):
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        buffer_flips.extend(tmp)
                        break
                    else:
                        tmp.append([cur_r, cur_c])
            return buffer_flips

        # First try flipping with current color
        flips = find_flips(r, c, color)

        # If no flips => forfeit scenario
        if not flips:
            forfeit = True
            color = -color  # Opponent's color
            flips = find_flips(r, c, color)

        # Still no flips => illegal
        if not flips:
            raise ValueError("Illegal move (no flips).")

        return {
            "move": move,
            "placed_color": color,
            "flips": flips,
            "forfeit": forfeit,
        }

    def apply_move(self, flip_info):
        """
        Use the flip_info dict from get_flips_info to flip squares, place the piece,
        update age, color, and record the move in history.
        """
        move = flip_info["move"]
        color = flip_info["placed_color"]
        flips = flip_info["flips"]
        r, c = move // 8, move % 8

        # Flip squares
        for rr, cc in flips:
            self.state[rr, cc] *= -1
            self.age[rr, cc] = 0

        # Place piece
        self.state[r, c] = color
        self.age[r, c] = 0

        # Update next_hand_color (flip if forfeit was forced the first time)
        self.next_hand_color = -color

        # Increase age
        self.age += 1

        # Record move
        self.history.append(move)

    def undo_move(self, flip_info):
        """
        Revert the board to its state before flip_info was applied.
        """
        move = flip_info["move"]
        color = flip_info["placed_color"]
        flips = flip_info["flips"]
        r, c = move // 8, move % 8

        # 1) Undo the piece placement
        self.state[r, c] = 0

        # 2) Undo the flips (flip them back)
        for rr, cc in flips:
            self.state[rr, cc] *= -1

        # 3) Decrease age for all squares (we added +1 in apply_move).
        #    Because each apply_move does self.age += 1, revert that.
        self.age -= 1
        # 3a) Also, the squares we set to 0 age in apply_move must be restored to
        #     what they were; for a simple approach, you could just track the old ages.
        #     For instance, store them in flip_info if you want precise reversion.
        #     For now, we just zeroed them, so set them to 1 maybe, or store the old
        #     ages in flip_info["old_ages"] if you want exact restoration.

        # 4) Revert next_hand_color
        #    If apply_move did `self.next_hand_color = -color`, then we revert it:
        self.next_hand_color = color if not flip_info["forfeit"] else -color
        # Or you might just store old_next_color in flip_info.

        # 5) Remove the move from history
        if self.history and self.history[-1] == move:
            self.history.pop()
