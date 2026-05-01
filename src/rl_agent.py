# src/rl_agent.py
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GRID_COLS, GRID_ROWS, Q_TABLE_PATH

# ── Constants ──────────────────────────────────────────────────
ACTIONS      = ["LEFT", "RIGHT", "FORWARD", "BACKWARD", "COLLECT"]
NUM_ACTIONS  = len(ACTIONS)
NUM_STATES   = GRID_ROWS * GRID_COLS + 1  # +1 for "no trash" state
NO_TRASH_STATE = NUM_STATES - 1

# ── Q-Learning Settings ────────────────────────────────────────
ALPHA        = 0.1    # learning rate — how fast robot learns
GAMMA        = 0.9    # discount — how much future reward matters
EPSILON      = 0.2    # exploration — chance of random action (20%)

class RLAgent:
    def __init__(self):
        """Initialize agent — load Q-table if exists, else create new."""
        self.actions    = ACTIONS
        self.alpha      = ALPHA
        self.gamma      = GAMMA
        self.epsilon    = EPSILON
        self.q_table    = self._load_q_table()
        self.prev_state  = None
        self.prev_action = None
        self.total_reward = 0
        self.step_count   = 0
        print(f"✅ RL Agent ready | States={NUM_STATES} Actions={NUM_ACTIONS}")
        print(f"   Q-table shape: {self.q_table.shape}")

    # ── Q-Table ────────────────────────────────────────────────
    def _load_q_table(self):
        """Load existing Q-table or create a new one."""
        if os.path.exists(Q_TABLE_PATH):
            q = np.load(Q_TABLE_PATH)
            print(f"✅ Q-table loaded from {Q_TABLE_PATH}")
            return q
        else:
            print(f"ℹ️  New Q-table created (no saved table found)")
            return np.zeros((NUM_STATES, NUM_ACTIONS))

    def save_q_table(self):
        """Save Q-table to disk so robot remembers between sessions."""
        os.makedirs(os.path.dirname(Q_TABLE_PATH), exist_ok=True)
        np.save(Q_TABLE_PATH, self.q_table)

    # ── State ──────────────────────────────────────────────────
    def get_state(self, detections, frame_width, frame_height):
        """
        Convert detection coordinates into a grid state.

        Frame is divided into a 3x3 grid:
        [0,1,2]
        [3,4,5]
        [6,7,8]

        If no trash → state = NO_TRASH_STATE
        If trash found → state = grid cell of closest trash
        """
        if not detections:
            return NO_TRASH_STATE

        # Find the closest trash object (smallest distance from center)
        frame_cx = frame_width  // 2
        frame_cy = frame_height // 2

        closest = min(detections,
                      key=lambda d: abs(d["center"][0] - frame_cx)
                                  + abs(d["center"][1] - frame_cy))

        cx, cy = closest["center"]

        # Map pixel position to grid cell
        col = min(int(cx / frame_width  * GRID_COLS), GRID_COLS - 1)
        row = min(int(cy / frame_height * GRID_ROWS), GRID_ROWS - 1)

        state = row * GRID_COLS + col
        return state

    # ── Action ─────────────────────────────────────────────────
    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        20% of the time → random action (explore)
        80% of the time → best known action (exploit)
        """
        if np.random.random() < self.epsilon:
            # Explore — try something random
            action_idx = np.random.randint(NUM_ACTIONS)
        else:
            # Exploit — use best known action
            action_idx = np.argmax(self.q_table[state])

        return action_idx, ACTIONS[action_idx]

    # ── Reward ─────────────────────────────────────────────────
    def get_reward(self, state, action_idx, detections, frame_width, frame_height):
        """
        Reward function — teaches robot what is good and bad.

        +10  → COLLECT when trash is in center cell (best!)
        +3   → Moving toward trash
        -1   → Moving away from trash
        -5   → COLLECT when no trash nearby (wasted action)
        0    → No trash visible
        """
        if state == NO_TRASH_STATE:
            return 0  # nothing to do

        action = ACTIONS[action_idx]
        center_state = 4  # center cell of 3x3 grid

        # COLLECT action
        if action == "COLLECT":
            if state == center_state:
                return +10  # perfect — trash is right in front
            else:
                return -5   # trash not close enough

        # Movement rewards — is robot moving toward center?
        col = state % GRID_COLS
        row = state // GRID_COLS

        # Distance from center cell
        dist = abs(col - 1) + abs(row - 1)

        if action == "LEFT"     and col > 1: return +3
        if action == "RIGHT"    and col < 1: return +3
        if action == "FORWARD"  and row > 1: return +3
        if action == "BACKWARD" and row < 1: return +3

        return -1  # moved away from trash

    # ── Learn ──────────────────────────────────────────────────
    def update_q_table(self, state, action_idx, reward, next_state):
        """
        Q-learning update formula:
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s')) - Q(s,a)]

        This is how the robot learns from experience.
        """
        current_q  = self.q_table[state, action_idx]
        max_next_q = np.max(self.q_table[next_state])
        new_q      = current_q + self.alpha * (
                         reward + self.gamma * max_next_q - current_q
                     )
        self.q_table[state, action_idx] = new_q

    # ── Main Step ──────────────────────────────────────────────
    def step(self, detections, frame_width, frame_height):
        """
        One full RL step:
        1. Get current state from detections
        2. Choose action
        3. Calculate reward
        4. Update Q-table
        5. Return action for robot to execute

        Returns: action name (string), state info (dict)
        """
        self.step_count += 1

        # Get current state
        state = self.get_state(detections, frame_width, frame_height)

        # Choose action
        action_idx, action_name = self.choose_action(state)

        # Get reward
        reward = self.get_reward(state, action_idx, detections,
                                 frame_width, frame_height)
        self.total_reward += reward

        # Update Q-table using previous step
        if self.prev_state is not None:
            self.update_q_table(
                self.prev_state,
                self.prev_action,
                reward,
                state
            )

        # Save current as previous for next step
        self.prev_state  = state
        self.prev_action = action_idx

        # Auto-save Q-table every 100 steps
        if self.step_count % 100 == 0:
            self.save_q_table()

        # Build info dictionary for dashboard
        info = {
            "state"       : state,
            "state_label" : self._state_label(state),
            "action"      : action_name,
            "reward"      : reward,
            "total_reward": round(self.total_reward, 1),
            "step"        : self.step_count
        }

        return action_name, info

    # ── Helpers ────────────────────────────────────────────────
    def _state_label(self, state):
        """Convert state number to human-readable position."""
        if state == NO_TRASH_STATE:
            return "No Trash"
        labels = [
            "Top-Left",    "Top-Center",    "Top-Right",
            "Mid-Left",    "Center",        "Mid-Right",
            "Bottom-Left", "Bottom-Center", "Bottom-Right"
        ]
        return labels[state] if state < len(labels) else f"State-{state}"


# ── Quick Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  RL AGENT TEST")
    print("=" * 50)

    agent = RLAgent()

    # Simulate 5 steps with fake detections
    fake_detections = [
        {"class_name": "bottle", "confidence": 0.85,
         "center": (400, 300), "bbox": (350, 250, 450, 350)}
    ]

    print("\nSimulating 5 decision steps:\n")
    for i in range(5):
        action, info = agent.step(fake_detections, 640, 480)
        print(f"  Step {i+1}:")
        print(f"    Trash position : {info['state_label']}")
        print(f"    Action decided : {info['action']}")
        print(f"    Reward         : {info['reward']}")
        print(f"    Total reward   : {info['total_reward']}")
        print()

    agent.save_q_table()
    print(f"✅ Q-table saved to logs/")
    print("\n" + "=" * 50)
    print("  RL AGENT WORKING — Ready for Step 8!")
    print("=" * 50)