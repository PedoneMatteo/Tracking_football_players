import sys
sys.path.append('../')
import cv2


class OffsideDetector:
    def __init__(self):
        self.attacking_direction_to_detect = False # if True, detect attacking direction from early frames (independent of KMeans) -> more robust to Team1/Team2 label swaps across runs. If False, relies on self.team_attacks_right, which is static.
        self.team_attacks_right = True
        
        
    def add_offside_to_tracks(self, tracks, trapezoids):
        """
        Adds 'offside' field (True/False) to every attacking player
        for each frame where the team has ball possession.

        Rules:
          - Team with smaller mean X = defends left goal → attacks towards the right.
          - Goalkeeper in frame (even outside trapezoid) → second-to-last defender.
          - Goalkeeper absent → last defender.
          - No exception for own half of the pitch.
        """
        # Detect attacking direction from early frames (independent of KMeans)
        prev_attacking_team = None
        attacking_team = 2
        #team_attacks_right = self._detect_attacking_direction(self.attacking_direction_to_detect, tracks)
        for frame_num, player_track in enumerate(tracks["players"]):
            # --- Step 1: detect goalkeeper presence in frame (even without position_transformed) ---
            gk_by_team = {1: False, 2: False}
            for track_id, track_info in player_track.items():
                team = track_info.get("team")
                if team is not None and track_info.get("is_goalkeeper", False):
                    gk_by_team[team] = True

            # --- Step 2: collect only players with valid position (inside trapezoid) ---
            team1_players = []
            team2_players = []

            for track_id, track_info in player_track.items():
                team = track_info.get("team")
                team_color = track_info.get("team_color")
                pos = track_info.get("position_transformed")
                if team is None or pos is None:
                    continue

                entry = {
                    "track_id": track_id,
                    "team": team,
                    "team_color": team_color,
                    "x": pos[0],   # coordinate along the pitch (meters)
                    "is_goalkeeper": track_info.get("is_goalkeeper", False),
                }

                if team == 1:
                    team1_players.append(entry)
                elif team == 2:
                    team2_players.append(entry)
 
            # --- Step 3: attacking team = whoever has the ball ---

            for track_id, track_info in player_track.items():
                if track_info.get("has_ball"):
                    attacking_team = track_info.get("team")
                    break

            # Skip frames without ball possessor
            if attacking_team is None:
                attacking_team = prev_attacking_team
                
            prev_attacking_team = attacking_team
            # --- Step 4: separate attackers/defenders ---
            attacks_right = True
            defending_team = 2 if attacking_team == 1 else 1

            if attacking_team == 1:
                attacking_players = team1_players
                defending_players = team2_players
            else:
                attacking_players = team2_players
                defending_players = team1_players

            gk_in_frame = gk_by_team[defending_team]

            # --- Step 5: compute offside line ---
            offside_line = self._get_offside_line(defending_players, attacks_right, gk_in_frame)
            if offside_line is None:
                continue

            # --- Step 6: mark offside for each attacker beyond the line ---
            for entry in attacking_players:
                if attacks_right:
                    is_offside = entry["x"] > offside_line
                else:
                    is_offside = entry["x"] < offside_line
                tracks["players"][frame_num][entry["track_id"]]["offside"] = is_offside

    def _detect_attacking_direction(self, direction_to_detect, tracks, sample_frames=30):
        """
        Detect attacking direction from average X positions in first sample_frames.
        Team with smaller mean X = on the left → attacks towards the right (+X).

        Needed because TeamAssigner uses non-deterministic KMeans → Team1/Team2
        labels may swap across runs.
        """
        if direction_to_detect:
            team_avg_x = {1: [], 2: []}
            for fnum, player_track in enumerate(tracks["players"]):
                if fnum >= sample_frames:
                    break
                for track_id, track_info in player_track.items():
                    team = track_info.get("team")
                    pos = track_info.get("position_transformed")
                    if team is not None and pos is not None:
                        team_avg_x[team].append(pos[0])

            avg_x = {}
            for team in [1, 2]:
                if team_avg_x[team]:
                    avg_x[team] = sum(team_avg_x[team]) / len(team_avg_x[team])
                else:
                    avg_x[team] = 0

            # Team on the left (smaller X) attacks towards the right
            team_attacks_right = {}
            if avg_x[1] <= avg_x[2]:
                team_attacks_right[1] = True
                team_attacks_right[2] = False
            else:
                team_attacks_right[1] = False
                team_attacks_right[2] = True

            return team_attacks_right
        else:
            # Static: Team1 attacks right, Team2 attacks left
            if self.team_attacks_right:
                return {1: True, 2: False}
            else:
                return {1: False, 2: True}
            

    def _get_offside_line(self, defending_players, attacks_right, gk_in_frame):
        """
        Compute the X coordinate of the offside line.

        Defender ordering:
          - attacks_right = True  → goal defended on the left → sort X descending
          - attacks_right = False → goal defended on the right → sort X ascending
        After ordering: sorted[0] = last defender (closest to goal),
                        sorted[1] = second-to-last defender.

        Rules:
          - Goalkeeper in frame  → line = second-to-last defender (sorted[1])
          - No goalkeeper, < 11  → line = last defender (sorted[0])
          - No goalkeeper, >= 11 → line = second-to-last defender (sorted[1])

        Returns None if there are no defenders.
        """
        if not defending_players:
            return None

        # Define sort order based on goal position
        if attacks_right:
            def sort_key(p): return -p["x"]
        else:
            def sort_key(p): return p["x"]

        sorted_players = sorted(defending_players, key=sort_key)
        visible_count = len(sorted_players)

        if gk_in_frame:
            # Standard case: goalkeeper visible → second-to-last defender
            if visible_count >= 2:
                return sorted_players[1]["x"]
            elif visible_count == 1:
                return sorted_players[0]["x"]
            return None
        else:
            # Goalkeeper not visible → last defender
            if visible_count >= 1:
                return sorted_players[0]["x"]
            return None
           
    def draw_offside(self, frames, tracks):
        """
        Draw a red border + "OFFSIDE" label on offside players.
        """
        for frame_num, frame in enumerate(frames):
            player_track = tracks["players"][frame_num]

            for track_id, track_info in player_track.items():
                if not track_info.get("offside"):
                    continue

                # Thick red border around the player
                bbox = track_info["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # "OFFSIDE" label above bbox (solid red background, white text)
                label = "OFFSIDE"
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_h - 8),
                    (x1 + text_w + 8, y1),
                    (0, 0, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        return frames
