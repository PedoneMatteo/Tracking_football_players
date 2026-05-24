import sys
sys.path.append('../')
import cv2


class OffsideDetector:
    def __init__(self):
        self.attacking_direction_to_detect = False # se True, rileva direzione attacco dai primi frame (indipendente da KMeans) -> più robusto a variazioni Team1/Team2 tra run diverse. Se False, dipende dal valore di self.team_attacks_right, che è statico.
        self.team_attacks_right = True
        
        
    def add_offside_to_tracks(self, tracks, trapezoids):
        """
        Aggiunge il campo 'offside' (True/False) a ogni giocatore in attacco
        per ogni frame in cui la squadra ha il possesso palla.

        Regole:
          - Squadra con X medio minore = difende porta sinistra → attacca verso destra.
          - Portiere in frame (anche fuori trapezoid) → penultimo difensore.
          - Portiere assente → ultimo difensore.
          - Nessuna eccezione per la propria metà campo.
        """
        # Determina direzione attacco dai primi frame (indipendente da KMeans)
        prev_attacking_team = None
        attacking_team = 2
        #team_attacks_right = self._detect_attacking_direction(self.attacking_direction_to_detect, tracks)
        for frame_num, player_track in enumerate(tracks["players"]):
            # --- Passo 1: rileva presenza portiere in frame (anche senza position_transformed) ---
            gk_by_team = {1: False, 2: False}
            for track_id, track_info in player_track.items():
                team = track_info.get("team")
                if team is not None and track_info.get("is_goalkeeper", False):
                    gk_by_team[team] = True
            print("\nFrame %d: GK squadra 1 = %s, GK squadra 2 = %s" % (frame_num, gk_by_team[1], gk_by_team[2]))
            
            # --- Passo 2: raccogli solo giocatori con posizione valida (dentro trapezoid) ---
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
                    "x": pos[0],   # coordinata lungo il campo (metri)
                    "is_goalkeeper": track_info.get("is_goalkeeper", False),
                }

                if team == 1:
                    team1_players.append(entry)
                elif team == 2:
                    team2_players.append(entry)

            # --- Passo 3: squadra in attacco = chi ha la palla ---
            
            for track_id, track_info in player_track.items():
                if track_info.get("has_ball"):
                    attacking_team = track_info.get("team")
                    break
            print("Squadra in attacco:", attacking_team)

            # Salta frame senza possessore palla
            if attacking_team is None:
                attacking_team = prev_attacking_team
                print("Nessun possessore palla rilevato, mantengo squadra in attacco precedente:", attacking_team)
                
            prev_attacking_team = attacking_team
            # --- Passo 4: separa attaccanti/difensori ---
            attacks_right = True
            defending_team = 2 if attacking_team == 1 else 1
            print("Squadra in difesa:", defending_team)

            if attacking_team == 1:
                attacking_players = team1_players
                defending_players = team2_players
            else:
                attacking_players = team2_players
                defending_players = team1_players

            gk_in_frame = gk_by_team[defending_team]

            # --- Passo 5: calcola linea di fuorigioco ---
            offside_line = self._get_offside_line(defending_players, attacks_right, gk_in_frame)
            print("Linea di fuorigioco (coordinata X in metri):", offside_line)
            if offside_line is None:
                continue

            # --- Passo 6: marca fuorigioco per ogni attaccante oltre la linea ---
            for entry in attacking_players:
                if attacks_right:
                    is_offside = entry["x"] > offside_line
                else:
                    is_offside = entry["x"] < offside_line
                if is_offside:
                    print("Frame %d: Giocatore %d (X = %.2f), squadra %s colore %s, è fuori gioco: %s" % (frame_num, entry["track_id"], entry["x"], entry["team"], entry["team_color"], is_offside))
                tracks["players"][frame_num][entry["track_id"]]["offside"] = is_offside
            print("\n ---------------------------------------------\n")

    def _detect_attacking_direction(self, direction_to_detect, tracks, sample_frames=30):
        """
        Determina direzione attacco da posizioni medie X nei primi sample_frames.
        Squadra con X medio minore = a sinistra → attacca verso destra (+X).

        Necessaria perché TeamAssigner usa KMeans non deterministico → label
        Team1/Team2 scambiate tra run diverse.
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

            # Squadra a sinistra (X minore) attacca verso destra
            team_attacks_right = {}
            if avg_x[1] <= avg_x[2]:
                team_attacks_right[1] = True
                team_attacks_right[2] = False
            else:
                team_attacks_right[1] = False
                team_attacks_right[2] = True

            return team_attacks_right
        else:
            # Statico: Team1 attacca verso destra, Team2 verso sinistra
            if self.team_attacks_right:
                return {1: True, 2: False}
            else:
                return {1: False, 2: True}
            

    def _get_offside_line(self, defending_players, attacks_right, gk_in_frame):
        """
        Calcola la coordinata X della linea di fuorigioco.

        Ordinamento difensori:
          - attacks_right = True  → porta difesa a sinistra → ordina X crescente
          - attacks_right = False → porta difesa a destra  → ordina X decrescente
        Dopo ordinamento: sorted[0] = ultimo difensore (più vicino alla porta),
                          sorted[1] = penultimo difensore.

        Regole:
          - Portiere in frame  → linea = penultimo difensore (sorted[1])
          - No portiere, < 11  → linea = ultimo difensore (sorted[0])
          - No portiere, >= 11 → linea = penultimo difensore (sorted[1])

        Restituisce None se non ci sono difensori.
        """
        if not defending_players:
            return None

        # Definisce ordinamento in base a dove si trova la porta
        if attacks_right:
            def sort_key(p): return -p["x"]
        else:
            def sort_key(p): return p["x"]

        sorted_players = sorted(defending_players, key=sort_key)
        visible_count = len(sorted_players)

        if gk_in_frame:
            # Caso standard: portiere visibile → penultimo difensore
            if visible_count >= 2:
                return sorted_players[1]["x"]
            elif visible_count == 1:
                return sorted_players[0]["x"]
            return None
        else:
            # Portiere non visibile → ultimo difensore
            if visible_count >= 1:
                return sorted_players[0]["x"]
            return None
           
    def draw_offside(self, frames, tracks):
        """
        Disegna un bordo rosso + etichetta "OFFSIDE" sui giocatori in fuorigioco.
        """
        for frame_num, frame in enumerate(frames):
            player_track = tracks["players"][frame_num]

            for track_id, track_info in player_track.items():
                if not track_info.get("offside"):
                    continue

                # Bordo rosso spesso attorno al giocatore
                bbox = track_info["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # Etichetta "OFFSIDE" sopra il bbox (sfondo rosso pieno, testo bianco)
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
