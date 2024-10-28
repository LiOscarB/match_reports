
import ScraperFC as sfc
import streamlit as st
import pandas as pd
import json
import requests
import ollama
import os

class Scraper:
    def __init__(self, ss_url: str, us_url: str):
        """
        Initialize the Scraper object with a URL and instances of the Sofascore and Understat classes.

        :param url: str - The URL of the match to scrape data from.
        """
        self.ss_url = ss_url
        self.us_url = us_url
        self.ss = sfc.Sofascore()
        self.us = sfc.Understat()
    
    def get_player_stats(self) -> pd.DataFrame:
        """
        Scrape player statistics from the provided URL using the Sofascore instance.

        :return: DataFrame containing individual player statistics with duplicate columns removed.
        """
        stats_df = self.ss.scrape_player_match_stats(self.ss_url)
        
        # Remove duplicate columns
        stats_df = stats_df.loc[:,~stats_df.columns.duplicated()].copy()
        return stats_df
    
    def get_momentum(self) -> pd.DataFrame:
        """
        Scrape momentum data from the match URL using Sofascore.
        Positive = momentum towards home team, Negative = momentum towards away team.

        :return: DataFrame containing momentum data with momentum at a given time interval.
        """
        momentum_df = self.ss.scrape_match_momentum(self.ss_url)
        return momentum_df
    
    def get_team_stats(self) -> pd.DataFrame:
        """
        Scrape team statistics from the match URL using Sofascore.

        :return: DataFrame containing team statistics.
        """
        team_stats_df = self.ss.scrape_team_match_stats(self.ss_url)
        return team_stats_df
    
    def get_match_stats(self) -> dict:
        """
        Retrieve match statistics as a dictionary from the match URL using Sofascore.

        :return: Dictionary containing general match statistics and game/team info.
        """
        
        # TO DO: REMOVE NON-NOTEWORTHY STATS
        match_stats_dict = self.ss.get_match_dict(self.ss_url)
        return match_stats_dict   
    
    def get_understat_match(self) -> tuple:
        """
        Scrape match data from a static Understat URL (specific match).

        :return: Dictionary containing Understat data for the specified match.
        
        In form of ({h/a key events}, {general match info}, {player stats})
        """
        understat_tuple = self.us.scrape_match(self.us_url)
        return understat_tuple
    
    def get_season_data(self) -> dict:
        # table = self.us.scrape_league_tables("2024/2025", "EPL")
        season_dict = self.us.scrape_all_teams_data("2024/2025", "EPL", as_df=True)

        return season_dict

class Preprocessor:
    def __init__(self):
        """
        Initializes the Preprocessor class, though currently no initialization logic is required.
        """
        pass

    def general_stats_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the general player stats DataFrame by removing irrelevant columns and formatting others.

        :param df: DataFrame - The DataFrame to preprocess.
        """
        # Initial drop of irrelevant columns
        df.drop(["firstName", "lastName", "userCount", "marketValueCurrency",
                 "dateOfBirthTimestamp", "fieldTranslations", "jerseyNumber",
                 ], axis=1, inplace=True)
        
        # Format country to only keep name of country:
        df['country_name'] = df['country'].apply(self.extract_country_name)
        # Drop the original column
        df = df.drop(columns=['country'])
        
        # Format match rating
        df['match_rating'] = df['ratingVersions'].apply(self.extract_rating)
        # Drop the original column
        df = df.drop(columns=['ratingVersions'])
        return df
        
    def preprocess_gk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the DataFrame to filter and clean data specific to goalkeepers.

        :param df: DataFrame - The DataFrame containing raw player stats.
        :return: DataFrame - The preprocessed DataFrame for goalkeepers.
        """
        gk_df = df.loc[df["position"] == "G"]
        try:
            gk_df.drop(["totalCross", "shotOffTarget", "onTargetScoringAttempt", "expectedGoals",
                        "bigChanceMissed", "outfielderBlock", "accurateCross", "goals","totalOffside"],
                        axis=1, inplace=True)
        except: 
            pass
        return gk_df
    
    def preprocess_outfield(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the DataFrame to filter and clean data specific to outfield players.

        :param df: DataFrame - The DataFrame of all player statistics player.
        :return: DataFrame - The preprocessed DataFrame for outfield players.
        """
        
        # Keep relevant outfield columns
        try:
            df.drop(["goodHighClaim", "savedShotsFromInsideTheBox", "saves", "punches",
                    "totalKeeperSweeper", "accurateKeeperSweeper", "goalsPrevented"], axis=1, inplace=True)
        except:
            pass
        df.drop(df[df.position == "GK"].index)
        return df
    
    def extract_country_name(self, json_dict: dict) -> str:
        """
        Extracts the country name from a JSON string.

        :param json_str: str - JSON string containing the country information.
        :return: str - The name of the country or None if the key is not found.
        """
        try:
            # Parse the JSON string into a dictionary
            # Return the rating
            return json_dict['name']
        except (json.JSONDecodeError, KeyError):
            # Return None or some default value if JSON is invalid or key is not found
            return None
    
    def extract_rating(self, json: dict) -> float:
        """
        Extracts the rating from a JSON string.

        :param json_str: str - JSON string containing rating information.
        :return: str - The extracted rating or None if the key is not found.
        """
        
        # If there is only one version of match rating, return it, else take the original rating
        if isinstance(json, float):
            return json
        elif isinstance(json, str):
            try:
                data_dict = json.loads(json)
                return float(data_dict['original'])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return None
        elif isinstance(json, dict):
            try:
                return float(json['original'])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return None
        else:
            return None
    
    def get_understat_events(self, understat_tuple: tuple) -> pd.DataFrame:
        events_dict = understat_tuple[0]
        # Normalize the data for home team
        h_events_df = pd.json_normalize(events_dict['h'])

        # Normalize the data for away team
        a_events_df = pd.json_normalize(events_dict['a'])

        
        # Concatenate both DataFrames
        events_df = pd.concat([h_events_df, a_events_df], ignore_index=True)

        # Preprocess xG, and filter out non-noteworth events
        events_df['xG'] = pd.to_numeric(events_df['xG'])
        events_df = events_df[(events_df["xG"] > 0.3) | (events_df["result"] == "Goal")]

        # preprocess shot type column
        
        events_df['shotType'] = events_df['shotType'].replace('leftFoot', 'left foot')
        events_df['shotType'] = events_df['shotType'].replace('rightFoot', 'right foot')
        
        # preprocess result column
        events_df['result'] =  events_df['result'].replace('BlockedShot', 'blocked')
        events_df['result'] =  events_df['result'].replace('SavedShot', 'saved')
        events_df['result'] =  events_df['result'].replace('Goal', 'a goal')
        
        events_df = events_df.drop(['id', 'match_id', 'h_goals', 'a_goals',
                                    'date', 'player_id', 'season'], axis=1)
        
        return events_df
    
    def get_team_season_data(self, events_df: pd.DataFrame, season_data: dict) -> pd.DataFrame:
        # TO DO: Last 5 opponents and (W/L)
        
        # Get home/away team names
        h_team_str = events_df.iloc[0]['h_team']
        a_team_str = events_df.iloc[0]['a_team']
        
        for team_str in [h_team_str, a_team_str]:
            team_key = team_str.replace(" ", "_")
            key = f"https://understat.com/team/{team_key}/2024"
            # Get record of previous games and put into a DataFrame
            record_df = season_data[key]['matches']
            record_df = record_df[record_df['isResult'] == True]
            this_gameweek = record_df[(record_df['h_title'] == h_team_str) & (record_df['a_title'] == a_team_str)]
            
            current_gameweek_date = this_gameweek['datetime'].iloc[0]
            # Filter to keep only records before the current game week
            previous_games_df = record_df[record_df['datetime'] < current_gameweek_date]
            
            if team_str is h_team_str:
                h_record = previous_games_df
                
                h_team = team_str
                # result, # W/L, H/A, H/A Record, # Last Games, # Games played
                h_result = this_gameweek['result'].iloc[0]
                if h_result == 'l':
                    h_num_wl = f"{h_record['result'].value_counts().get('l', 0) + 1} season losses"
                else: 
                    h_num_wl = f"{h_record['result'].value_counts().get('w', 0) + 1} season wins"
                
                # Select the last 5 or fewer results home games
                last_5_h = h_record[h_record['side'] == 'h']['result'].tail(5).tolist()
                last_5_h_str = '-'.join(last_5_h).upper()
                
                # Get last 5 games record
                h_last_5 = h_record['result'].tail(5).tolist()
                h_last_5_str = '-'.join(h_last_5).upper()
                h_games_played = len(previous_games_df) + 1
                   
            else:
                a_record = previous_games_df
                a_team = team_str
                a_result = this_gameweek['result'].iloc[0]
                if a_result == 'l':
                    a_num_wl = f"{a_record['result'].value_counts().get('l', 0) + 1} season losses"
                else: 
                    a_num_wl = f"{a_record['result'].value_counts().get('w', 0) + 1} season wins"
                
                # Select the last 5 or fewer results home games
                last_5_a = a_record[a_record['side'] == 'a']['result'].tail(5).tolist()
                last_5_a_str = '-'.join(last_5_a).upper()
                

                
                # Get last 5 games record
                a_last_5 = a_record['result'].tail(5).tolist()
                a_last_5_str = '-'.join(a_last_5).upper()
                
                a_games_played = len(previous_games_df) + 1
                
        team_prev_game_dict = {
        'team': [h_team, a_team],
        'h_a': ['home', 'away'],
        'result': [h_result, a_result],
        'num_wl': [h_num_wl, a_num_wl],
        'h_a_record': [last_5_h_str, last_5_a_str],
        'last_5': [h_last_5_str, a_last_5_str],
        'games_played': [h_games_played, a_games_played]
        }
        team_prev_game_df = pd.DataFrame(team_prev_game_dict)
        return team_prev_game_df
    
    def get_team_llm_prompt(self, game_stats: pd.DataFrame, match_dict: dict,
                            event_df: pd.DataFrame, team_prev_game_df: pd.DataFrame) -> str:
        """
        Constructs a detailed match report prompt based on game statistics and match information.

        :param game_stats: DataFrame - The DataFrame containing various game statistics from SofaScore.
        :param match_dict: dict - A dictionary containing general match information.
        :param event_df: DataFrame - DataFrame containing key events.
        :param team_prev_game_df: DataFrame - DataFrame containing previous form 
        
        :return: str - A string containing the formatted prompt for generating a match report.
        """
        
        # Get general match info
        comp = match_dict["tournament"]["name"]
        season = match_dict["season"]["year"]
        gw = match_dict["roundInfo"]["round"]
        home_team = match_dict["homeTeam"]["name"]
        h_manager = match_dict["homeTeam"]["manager"]["name"]
        away_team = match_dict["awayTeam"]["name"]
        a_manager = match_dict["awayTeam"]["manager"]["name"]
        stadium = match_dict["venue"]["stadium"]["name"]
        
        ht_h_score = match_dict["homeScore"]["period1"]
        ft_h_score = match_dict["homeScore"]["current"]
        
        ht_a_score = match_dict["awayScore"]["period1"]
        ft_a_score = match_dict["awayScore"]["current"]
        
        prompt = "Match Stats Summary: \n\n"
        
        # Pivot the data
        pivot_df = game_stats.pivot_table(index=['key', 'group'], columns='period', values=['home', 'away'], aggfunc='first')

        # Iterate and format strings
        for index, row in pivot_df.iterrows():
            event, category = index
            prompt += f"{event}: "
            for period in ['ALL', '1ST', '2ND']:
                home = row[('home', period)]
                away = row[('away', period)]
                # Check if both home and away data are not NaN
                if pd.notna(home) and pd.notna(away):
                    prompt += f"{period}: Home {home}, Away {away}; "
                elif pd.notna(home):  # Only home is not NaN
                    prompt += f"{period}: Home {home}; "
                elif pd.notna(away):  # Only away is not NaN
                    prompt += f"{period}: Away {away}; "
            prompt += f"\n\n"
        prompt += (f"Competition: {comp} - Season: {season} - Gameweek: {gw} at Stadium: {stadium}\n\n"
                f"Home Team: {home_team}, Home Team Manager: {h_manager}\n\n"
                f"Away Team: {away_team}, Away Team Manager: {a_manager}\n\n"
                f"half time score {home_team}: {ht_h_score}, {away_team}: {ht_a_score} "
                f"full time score {home_team}: {ft_h_score}, {away_team}: {ft_a_score}\n\n")

        prompt += "**Key events:** \n\n"
        
        home_events = event_df[event_df["h_a"] == 'h']
        away_events = event_df[event_df["h_a"] == 'a']
        for event in [home_events, away_events]:
            if event is home_events:
                prompt += f"**{home_team} key events:** \n\n"
            else:
                prompt += f"**{away_team} key events:** \n\n"
            for _, row in event.iterrows():
                description = f"**{row['minute']}th minute:** Player: {row['player']}, Shot type: {row['shotType'].lower()}, Outcome: {row['result'].lower()}, "
                if pd.notna(row['player_assisted']):
                    description += f"assisted by {row['player_assisted']}, "
                description += f"xG: {row['xG']}. \n\n"
                prompt += description
        
        prompt += "**Season Records:** \n\n"
        for index, team in team_prev_game_df.iterrows():
            prompt += f"{team['team']} now has {team['num_wl']}. last {len(team['h_a_record'].split('-'))} {team['h_a']} game(s): {team['h_a_record']}, last {len(team['last_5'].split('-'))} game(s): {team['last_5']}, games played: {team['games_played']} \n\n"
            
        prompt += (f"Using only the information provided, generate an interesting match report with"
                   f"the following three subheadings: \n"
                   f"Key Events: in chronological order (400 words). \n"
                   f"Match Stats Summary: detailed and in sentence format e.g. shots, fouls, big chances, passes, possession (minimum 400 words). \n"
                   f"Overview of season so far: using season records (100 words)."
                   f"There should be around 1000 words total. If there are extra words, use it in Match stats summary"
                   f"Note: There is currently no table data, so do not mention league tables, "
                   f"do not use dot points")
        
        # Note, need to add more data, momentum, standings? goalscorers
        return prompt

class LLM:
    """A class to interact with a language model API to generate text based on prompts."""
    def __init__(self):
        """Initialize the API URL and headers for JSON content type."""
        self.url = "http://127.0.0.1:11434/api/generate"
        self.headers = {"Content-Type": "application/json"}
        return

    def general_match_report(self, prompt: str) -> str:
        """
        Send a prompt to the language model API and return the generated text.

        :param prompt: str - The text prompt to send to the language model.
        :return: str - The generated report or an error message.
        """
        data = {
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        
        if response.status_code == 200:
            response_text = response.text
            data = json.loads(response_text)
            generated_report = data["response"]
        else: 
            generated_report = f"Error:, {response.status_code}, {response.text}, unluggy no report"
            
            
        try:
            # Make the POST request
            response = requests.post(self.url, headers=self.headers, json=data)

            # Check if the request was successful
            if response.status_code == 200:
                print("Successful response")
            else:
                print(f"Error: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            # Handle connection errors
            print(f"Connection error: {e}")
        return generated_report
    

class Streamlit:
    """Class for handling the visualization and presentation of data using Streamlit."""
    def __init__(self):
        pass
    
    def upload_stats(self, gk_stats: pd.DataFrame, o_stats: pd.DataFrame) -> None:
        """Display goalkeeper and outfield player statistics in Streamlit."""
        OUTFIELD_POSITIONS = ["D", "M", "F"]
        POSITION_NAMES = {"D": "Defenders", "M": "Midfielders",
                          "F": "Forwards"}
        
        # Upload GK Stats 
        st.header("Goalkeepers:")
        st.dataframe(gk_stats)
        
        # Upload outfield stats by position
        for pos in OUTFIELD_POSITIONS:
            pos_df = o_stats.loc[o_stats["position"] == pos]
            st.header(POSITION_NAMES[pos])
            st.dataframe(pos_df)
        return
    
    def upload_momentum(self, momentum_df: pd.DataFrame) -> None:
        """Display momentum data in Streamlit."""
        st.header("Momentum:")
        st.dataframe(momentum_df)
        return
    
    def upload_team_stats(self, team_df: pd.DataFrame) -> None:
        """Display team statistics in Streamlit."""
        st.header("Team Stats:")
        st.dataframe(team_df)
        return
    
    def upload_match_stats(self, match_dict: dict) -> None:
        """Display team statistics in Streamlit."""
        st.header("Match Dict:")
        st.write(match_dict)
        return
    
    def upload_match_report(self, prompt: str, output: str) -> None:
        """Display the generated match report in Streamlit."""
        st.header("Original Message:")
        st.write(prompt)
        st.header("Generated Report")
        st.write(output)
        return
    
    def upload_prompt(self, prompt: str):
        """Display original prompt in Streamlit."""
        st.header("Original Message:")
        st.write(prompt)
        return
    
    def upload_understat(self, understat_df: dict):
        """Display Understat match data in Streamlit."""
        st.header("Understat DF")
        st.write(understat_df)
    
class Orchestrator:
    """Class to orchestrate the scraping, processing, and display of match data."""
    def __init__(self):
        """Initialize with components for scraping, processing, and display."""
        ss_url = os.getenv('SS_URL', 'https://www.sofascore.com/football/match/liverpool-manchester-united/KU#id:12436920')
        us_url = os.getenv('US_URL', 'https://understat.com/match/26631')
        self.Scraper = Scraper(ss_url, us_url)
        self.Preprocessor = Preprocessor()
        self.Streamlit = Streamlit()
        self.LLM = LLM()
          
    def run(self):
        """Run the orchestration process to display match data and generated reports."""
        # player_stats_df = self.Preprocessor.general_stats_preprocess(self.Scraper.get_player_stats())
        # momentum_df = self.Scraper.get_momentum()
        team_stats_df = self.Scraper.get_team_stats()
        match_stats_dict = self.Scraper.get_match_stats()
        understat_tuple = self.Scraper.get_understat_match()
        season_data = self.Scraper.get_season_data()
        
        key_events_df = self.Preprocessor.get_understat_events(understat_tuple)
        team_prev_game_df = self.Preprocessor.get_team_season_data(key_events_df, season_data)
        
        # gk_stats = self.Preprocessor.preprocess_gk(player_stats_df)
        # o_stats = self.Preprocessor.preprocess_outfield(player_stats_df)
        prompt = self.Preprocessor.get_team_llm_prompt(game_stats = team_stats_df,
                                                       match_dict = match_stats_dict,
                                                       event_df = key_events_df,
                                                       team_prev_game_df = team_prev_game_df)
        llm_output = self.LLM.general_match_report(prompt)
        # self.Streamlit.upload_stats(gk_stats=gk_stats, o_stats=o_stats)
        # self.Streamlit.upload_momentum(momentum_df=momentum_df)
        #self.Streamlit.upload_team_stats(team_df=team_prev_game_df)
        self.Streamlit.upload_match_report(prompt, llm_output)
        #self.Streamlit.upload_understat(season_data)
        #self.Streamlit.upload_understat(understat_tuple)
        # self.Streamlit.upload_match_stats(match_dict=match_stats_dict)
        
        
     
if __name__ == "__main__": 
    orchestrator = Orchestrator()
    orchestrator.run()        

