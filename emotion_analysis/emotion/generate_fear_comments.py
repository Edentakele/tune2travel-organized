import pandas as pd
import random
import uuid
from datetime import datetime, timezone

# Enhanced list of fear-inducing keywords and phrases
FEAR_KEYWORDS = [
    "scared", "terrified", "frightened", "horror", "panic", "anxious", "nervous",
    "dread", "creepy", "eerie", "ghost", "monster", "nightmare", "dark", "shadows",
    "haunted", "danger", "warning", "help me", "trapped", "alone", "lost", "attack",
    "chase", "scream", "phobia", "paranoia", "threat", "spooky", "chilling",
    "if this happens", "what if", "I'm worried", "be careful", "watch out",
    "something bad is going to happen", "I have a bad feeling", "this is unsettling",
    "makes my skin crawl", "sends shivers down my spine", "heart pounding", "can't breathe",
    "shaking", "sweating", "jump scare", "ominous", "foreboding", "cursed", "possessed",
    "evil", "demon", "witch", "vampire", "zombie", "apocalypse", "survival", "end of the world",
    "unknown", "mystery", "unexplained", "supernatural", "paranormal", "glitch in the matrix",
    "simulation", "government conspiracy", "deep web", "dark net", "stalker", "kidnapped",
    "murder", "serial killer", "true crime", "don't look back", "it's behind you",
    "they're watching", "I know your secret", "you're next"
]

# Templates for generating fear comments, incorporating more variety and realism
FEAR_TEMPLATES = [
    "Just thinking about {topic} makes me {keyword}.",
    "I had a nightmare last night that {event_description}, it felt so real.",
    "What if {scenario}? That's a {keyword} thought.",
    "I'm actually {keyword} of {noun_phrase}.",
    "This song/video reminds me of that {keyword} {noun_phrase} I saw.",
    "Anyone else get a {keyword} feeling from {specific_detail}?",
    "I can't shake this {keyword} feeling that {potential_threat}.",
    "Is it just me, or is {observation} really {keyword}?",
    "The part where {specific_moment_in_media} happens is genuinely {keyword}.",
    "Reading about {topic} late at night... big mistake. So {keyword}!",
    "My biggest fear is {fear_description}. This is too close.",
    "This whole situation feels like a {keyword} movie plot.",
    "I keep hearing strange noises, I'm starting to get {keyword}.",
    "The thought of {abstract_fear} is enough to make me {keyword}.",
    "I'm not usually {keyword}, but {trigger} really got to me.",
    "That {sound_description} was so {keyword}, did anyone else hear it?",
    "This place has a really {keyword} vibe, let's get out of here.",
    "I have a phobia of {specific_phobia}, and this is triggering it.",
    "The way {character_action} was portrayed was deeply {keyword}.",
    "Seriously, {question_about_safety}? That's {keyword}.",
    "I'm {keyword} that {conspiracy_theory} might actually be true.",
    "Just saw something move in the {location}, I'm {keyword}!",
    "The silence is almost more {keyword} than the noise.",
    "I've got a bad feeling about this, like {ominous_comparison}.",
    "This is giving me major {keyword} flashbacks to {past_event}.",
    "Don't go into the {place}, it's {keyword} in there!",
    "I swear I just saw {supernatural_entity}, I'm so {keyword}.",
    "The {weather_condition} makes everything feel so {keyword}.",
    "What if {terrifying_what_if_scenario}? I'm legitimately {keyword}.",
    "This is the kind of thing that keeps me up at night, so {keyword}.",
    "I'm {keyword} about the {future_event}, it feels so uncertain.",
    "The {texture_or_smell} is really {keyword}, it's unsettling.",
    "I read a story about {urban_legend} and now I'm {keyword}.",
    "The way the {object} was {state_of_object} was so {keyword}.",
    "I'm {keyword} of being {state_of_being_feared}, it's my worst nightmare.",
    "This feels like a scene from a {horror_subgenre} film. So {keyword}!",
    "I can't stop thinking about {disturbing_thought}, it's making me {keyword}.",
    "The {lack_of_something} is what makes this so {keyword}.",
    "I'm {keyword} that {technology_related_fear}.",
    "This {artwork_or_music} evokes such a {keyword} feeling.",
    "I had a premonition that {bad_event_premonition}, it's making me {keyword}.",
    "The {historical_event_with_fear} is a reminder of how {keyword} things can get.",
    "I'm {keyword} of what's hiding in the {place_where_things_hide}.",
    "That {animal_or_creature} is my biggest {keyword}.",
    "The sudden {sensory_experience} was incredibly {keyword}.",
    "I'm {keyword} about what {person_or_group} will do next.",
    "This {news_story_element} is genuinely {keyword}.",
    "I feel like I'm being watched, it's so {keyword}.",
    "The {repeated_sound_or_action} is driving me crazy, it's so {keyword}.",
    "I'm {keyword} of {existential_fear} more than anything."
]

# More diverse elements for template filling
TOPICS = [
    "the dark", "heights", "spiders", "the future", "being alone", "public speaking",
    "failure", "the unknown", "what's under the bed", "that noise outside", "this dream I had",
    "AI taking over", "climate change", "a parallel universe", "losing someone", "deep water",
    "the government", "social media", "the news", "old buildings", "clowns", "empty hallways",
    "creaking doors", "whispers in the wind", "sudden shadows"
]

EVENT_DESCRIPTIONS = [
    "I was being chased by a shadowy figure", "all my teeth fell out",
    "I was trapped in a maze with no exit", "I couldn't scream for help",
    "the world was ending", "I saw a doppelganger of myself", "I was falling endlessly",
    "something was scratching at the door", "a face appeared in the window",
    "the floorboards were creaking upstairs"
]

SCENARIOS = [
    "the power goes out right now", "that was a real ghost", "we're actually in a simulation",
    "someone is listening to this conversation", "this is all a setup", "the prophecy is true",
    "we run out of resources", "another pandemic hits", "the internet suddenly dies",
    "all the mirrors show a different reflection"
]

NOUN_PHRASES = [
    "creature in the woods", "urban legend", "unsolved mystery", "glitch in reality",
    "feeling of being watched", "sound from the basement", "shadow in the corner of my eye",
    "prophetic dream", "cursed object", "abandoned place", "true crime story",
    "eerie silence", "flickering streetlight", "cold breath on my neck"
]

SPECIFIC_DETAILS = [
    "the way the eyes followed me", "that sudden silence", "the flickering lights",
    "the distorted face in the reflection", "the cold spot in the room", "the faint whisper",
    "the door creaking open by itself", "the object that moved on its own",
    "the way the temperature dropped", "the unexplained static on the radio"
]

POTENTIAL_THREATS = [
    "someone is in the house", "this isn't over", "they know where I am",
    "it's coming back for me", "I'm not safe here", "something terrible is about to unfold",
    "this is just the beginning", "we're all in danger", "it's right behind you",
    "you can't escape it"
]

OBSERVATIONS = [
    "how quiet it is tonight", "that figure in the distance", "the way the wind is howling",
    "this sudden change in atmosphere", "the number of coincidences lately",
    "how everyone is acting strangely", "the lack of birdsong", "this unexplainable phenomenon",
    "the unsettling stillness", "the unnatural fog rolling in"
]

FEAR_DESCRIPTIONS = [
    "being buried alive", "losing my mind", "never waking up from a nightmare",
    "seeing something I can't unsee", "the world ending and I'm the only one left",
    "being completely forgotten", "that something is living in my walls", "drowning",
    "being paralyzed while aware", "an unknown entity in my room"
]

CONSPIRACY_THEORIES = [
    "Area 51 holding aliens", "the moon landing being faked on a set", "Bigfoot sightings in national parks",
    "the Illuminati controlling global events", "chemtrails being used for mind control",
    "lizard people in positions of power", "the Earth actually being flat",
    "that we're all living in a computer simulation (NPCs)", "secret government experiments",
    "numbers stations broadcasting coded messages"
]

ABSTRACT_FEARS = [
    "the passage of time", "the concept of infinity", "losing my memories", "not existing",
    "the true nature of reality", "what lies beyond death", "being truly alone in the universe",
    "the meaninglessness of existence", "the unknown future", "my own potential for evil"
]

SPECIFIC_PHOBIAS = [
    "arachnophobia (spiders)", "ophidiophobia (snakes)", "acrophobia (heights)",
    "claustrophobia (enclosed spaces)", "trypophobia (holes)", "thanatophobia (death)",
    "glossophobia (public speaking)", "mysophobia (germs)", "cynophobia (dogs)", "astraphobia (thunder/lightning)"
]

FUTURE_EVENTS = [
    "the upcoming solar eclipse", "the next global election", "the release of that new AI",
    "my final exams", "the outcome of the current crisis", "what humanity will become",
    "the discovery of extraterrestrial life", "the next technological singularity"
]

def generate_comment_text():
    template = random.choice(FEAR_TEMPLATES)
    keyword = random.choice(FEAR_KEYWORDS)

    # Fill placeholders based on the template's needs
    comment = template.replace("{keyword}", keyword)
    if "{topic}" in comment:
        comment = comment.replace("{topic}", random.choice(TOPICS))
    if "{event_description}" in comment:
        comment = comment.replace("{event_description}", random.choice(EVENT_DESCRIPTIONS))
    if "{scenario}" in comment:
        comment = comment.replace("{scenario}", random.choice(SCENARIOS))
    if "{noun_phrase}" in comment:
        comment = comment.replace("{noun_phrase}", random.choice(NOUN_PHRASES))
    if "{specific_detail}" in comment:
        comment = comment.replace("{specific_detail}", random.choice(SPECIFIC_DETAILS))
    if "{potential_threat}" in comment:
        comment = comment.replace("{potential_threat}", random.choice(POTENTIAL_THREATS))
    if "{observation}" in comment:
        comment = comment.replace("{observation}", random.choice(OBSERVATIONS))
    if "{specific_moment_in_media}" in comment:
        comment = comment.replace("{specific_moment_in_media}", random.choice([
            "the doll's eyes moved", "the sudden jump scare", "the creature's reveal",
            "the eerie music crescendoed", "the character walked into the dark room"
        ]))
    if "{fear_description}" in comment:
        comment = comment.replace("{fear_description}", random.choice(FEAR_DESCRIPTIONS))
    if "{trigger}" in comment:
        comment = comment.replace("{trigger}", random.choice([
            "that sudden loud noise", "the power flickering", "seeing a spider",
            "being in a tight space", "the news report I just heard"
        ]))
    if "{sound_description}" in comment:
        comment = comment.replace("{sound_description}", random.choice([
            "creaking floorboard upstairs", "a faint whisper from the other room",
            "scratching inside the walls", "an unfamiliar footstep outside", "the wind howling like a wolf"
        ]))
    if "{character_action}" in comment:
        comment = comment.replace("{character_action}", random.choice([
            "the villain smiled slowly", "the child pointed at nothing",
            "the reflection didn't match", "the old man stared blankly", "the dog started growling at the empty corner"
        ]))
    if "{question_about_safety}" in comment:
        comment = comment.replace("{question_about_safety}", random.choice([
            "Did you lock the door?", "Are we alone here?", "What was that sound?",
            "Is someone following us?", "Should we really open that?"
        ]))
    if "{conspiracy_theory}" in comment:
        comment = comment.replace("{conspiracy_theory}", random.choice(CONSPIRACY_THEORIES))
    if "{location}" in comment:
        comment = comment.replace("{location}", random.choice([
            "dark hallway", "abandoned hospital", "creepy basement", "foggy forest", "empty attic"
        ]))
    if "{ominous_comparison}" in comment:
        comment = comment.replace("{ominous_comparison}", random.choice([
            "a storm is brewing", "the calm before the horror", "a predator stalking its prey",
            "a ticking time bomb", "the silence of a graveyard"
        ]))
    if "{past_event}" in comment:
        comment = comment.replace("{past_event}", random.choice([
            "that time I got lost in the woods", "the night the power went out for days",
            "when I thought I saw a ghost", "the scary movie I watched last week", "the local urban legend I heard"
        ]))
    if "{place}" in comment:
        comment = comment.replace("{place}", random.choice([
            "old attic", "dark woods", "abandoned house", "creepy cellar", "that locked room"
        ]))
    if "{supernatural_entity}" in comment:
        comment = comment.replace("{supernatural_entity}", random.choice([
            "a ghostly figure", "a shadowy demon", "a poltergeist", "a banshee", "something not human"
        ]))
    if "{weather_condition}" in comment:
        comment = comment.replace("{weather_condition}", random.choice([
            "thick fog", "pouring rain and thunder", "an unnatural silence in the air",
            "a blood moon", "howling winds"
        ]))
    if "{terrifying_what_if_scenario}" in comment:
        comment = comment.replace("{terrifying_what_if_scenario}", random.choice([
            "the power grid fails permanently", "we discover aliens and they're hostile",
            "a zombie apocalypse actually starts", "our deepest fears manifest physically", "time starts looping"
        ]))
    if "{future_event}" in comment:
        comment = comment.replace("{future_event}", random.choice(FUTURE_EVENTS))
    if "{texture_or_smell}" in comment:
        comment = comment.replace("{texture_or_smell}", random.choice([
            "damp, earthy smell", "the metallic scent of blood", "a cold, slimy touch",
            "the smell of decay", "an unidentifiable, sweet but rotten odor"
        ]))
    if "{urban_legend}" in comment:
        comment = comment.replace("{urban_legend}", random.choice([
            "Slender Man", "Bloody Mary", "the Mothman", "a local ghost story", "the vanishing hitchhiker"
        ]))
    if "{object}" in comment:
        comment = comment.replace("{object}", random.choice([
            "old doll", "antique mirror", "music box that plays by itself", "a dusty, ancient book", "a single, unblinking eye"
        ]))
    if "{state_of_object}" in comment:
        comment = comment.replace("{state_of_object}", random.choice([
            "slightly moved on its own", "staring right at me", "covered in a strange substance",
            "glowing faintly", "whispering my name"
        ]))
    if "{state_of_being_feared}" in comment:
        comment = comment.replace("{state_of_being_feared}", random.choice([
            "completely paralyzed", "utterly alone in the dark", "hunted by something unknown",
            "unable to scream", "lost in an endless maze"
        ]))
    if "{horror_subgenre}" in comment:
        comment = comment.replace("{horror_subgenre}", random.choice([
            "found footage", "psychological thriller", "slasher", "body horror", "cosmic horror"
        ]))
    if "{disturbing_thought}" in comment:
        comment = comment.replace("{disturbing_thought}", random.choice([
            "what happens after we die", "the idea that we're not in control",
            "the fragility of sanity", "the vastness of the universe and our insignificance", "that I might be the monster"
        ]))
    if "{lack_of_something}" in comment:
        comment = comment.replace("{lack_of_something}", random.choice([
            "sound in the woods", "reflection in the mirror", "escape route", "explanation for the events", "help"
        ]))
    if "{technology_related_fear}" in comment:
        comment = comment.replace("{technology_related_fear}", random.choice([
            "my webcam is hacked", "AI becomes sentient and hostile", "my smart home turns against me",
            "deepfakes are indistinguishable from reality", "my data is used to create a digital clone"
        ]))
    if "{artwork_or_music}" in comment:
        comment = comment.replace("{artwork_or_music}", random.choice([
            "painting with watching eyes", "a song that seems to predict the future",
            "a sculpture that moves when you're not looking", "a discordant melody that causes unease", "a children's rhyme with dark undertones"
        ]))
    if "{bad_event_premonition}" in comment:
        comment = comment.replace("{bad_event_premonition}", random.choice([
            "a car crash", "a loved one's death", "a natural disaster", "a personal failure", "a global catastrophe"
        ]))
    if "{historical_event_with_fear}" in comment:
        comment = comment.replace("{historical_event_with_fear}", random.choice([
            "the Black Plague", "the Salem Witch Trials", "Jack the Ripper's reign", "the World Wars", "the Chernobyl disaster"
        ]))
    if "{place_where_things_hide}" in comment:
        comment = comment.replace("{place_where_things_hide}", random.choice([
            "shadows", "under the bed", "in the closet", "just out of sight", "in your own mind"
        ]))
    if "{animal_or_creature}" in comment:
        comment = comment.replace("{animal_or_creature}", random.choice([
            "giant centipede", "a swarm of spiders", "a snake in the grass", "a wolf with glowing eyes", "something with too many teeth"
        ]))
    if "{sensory_experience}" in comment:
        comment = comment.replace("{sensory_experience}", random.choice([
            "cold breath on my neck", "a touch in the dark", "a whisper of my name",
            "the smell of sulfur", "a flash of movement in my peripheral vision"
        ]))
    if "{person_or_group}" in comment:
        comment = comment.replace("{person_or_group}", random.choice([
            "that strange cult", "the government", "the neighbors who never come out",
            "the man in the long coat", "the eyes watching from the window"
        ]))
    if "{news_story_element}" in comment:
        comment = comment.replace("{news_story_element}", random.choice([
            "unexplained disappearances", "a new, deadly virus", "strange sightings in the sky",
            "a series of unsolved murders", "reports of mass hysteria"
        ]))
    if "{repeated_sound_or_action}" in comment:
        comment = comment.replace("{repeated_sound_or_action}", random.choice([
            "tapping on the window", "a floorboard creaking at the same time each night",
            "a distant, repetitive scream", "a child's laughter where no child should be", "the same dream every night"
        ]))
    if "{existential_fear}" in comment:
        comment = comment.replace("{existential_fear}", random.choice(ABSTRACT_FEARS))
    if "{abstract_fear}" in comment:
        comment = comment.replace("{abstract_fear}", random.choice(ABSTRACT_FEARS))
    if "{specific_phobia}" in comment:
        comment = comment.replace("{specific_phobia}", random.choice(SPECIFIC_PHOBIAS))

    # Ensure all placeholders are filled, if any are missed, fill with a generic keyword
    final_comment = comment
    if "{" in final_comment and "}" in final_comment:
        # A simple way to replace any remaining placeholders
        final_comment = final_comment.replace(final_comment[final_comment.find("{"):final_comment.find("}")+1], random.choice(FEAR_KEYWORDS))

    return final_comment.capitalize()

def load_used_comment_ids(file_paths):
    used_ids = set()
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, usecols=['comment_id'])
            used_ids.update(df['comment_id'].tolist())
        except FileNotFoundError:
            print(f"Warning: Manually labelled file not found: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return used_ids

def get_unused_comment_metadata(no_spam_file_paths, used_comment_ids, num_needed, all_original_columns):
    unused_metadata = []
    required_cols = ['comment_id', 'author_name', 'published_at', 'published_at_unix', 'like_count'] # Core metadata
    
    available_cols_set = set(all_original_columns)
    cols_to_load = [col for col in required_cols if col in available_cols_set]
    # Add original_comment temporarily to ensure we only pick English comments, then drop
    cols_to_load_for_filtering = cols_to_load + ['original_comment', 'detected_language']

    for file_path in no_spam_file_paths:
        if len(unused_metadata) >= num_needed:
            break
        try:
            # Read in chunks to handle large files
            chunk_iter = pd.read_csv(
                file_path, 
                usecols=lambda x: x in cols_to_load_for_filtering or x in all_original_columns, # Load necessary + original
                chunksize=10000, 
                low_memory=False
            )
            for chunk in chunk_iter:
                # Filter for English comments not already used
                potential_comments = chunk[
                    (~chunk['comment_id'].isin(used_comment_ids)) & 
                    (chunk['detected_language'] == 'en')
                ]
                for _, row in potential_comments.iterrows():
                    meta = {col: row.get(col) for col in all_original_columns} # Get all original cols
                    meta['comment_id'] = row.get('comment_id', f"fallbackid_{uuid.uuid4()}")
                    meta['author_name'] = row.get('author_name', f"AnonUser_{random.randint(1000,9999)}")
                    
                    # Ensure published_at is valid or generate a new one
                    published_at_val = row.get('published_at')
                    try:
                        # Attempt to parse to ensure it's a valid ISO format datetime
                        datetime.fromisoformat(str(published_at_val).replace('Z', '+00:00'))
                        meta['published_at'] = published_at_val
                    except (ValueError, TypeError):
                        fallback_dt = datetime.now(timezone.utc) - pd.Timedelta(days=random.randint(1, 365))
                        meta['published_at'] = fallback_dt.isoformat().replace('+00:00', 'Z')

                    # Ensure published_at_unix is valid or derive/generate
                    published_at_unix_val = row.get('published_at_unix')
                    if pd.isna(published_at_unix_val):
                        try:
                            meta['published_at_unix'] = int(datetime.fromisoformat(str(meta['published_at']).replace('Z', '+00:00')).timestamp())
                        except (ValueError, TypeError):
                             meta['published_at_unix'] = int((datetime.now(timezone.utc) - pd.Timedelta(days=random.randint(1,365))).timestamp())
                    else:
                         meta['published_at_unix'] = int(published_at_unix_val)

                    meta['like_count'] = int(row.get('like_count', 0)) if pd.notna(row.get('like_count')) else 0
                    
                    unused_metadata.append(meta)
                    if len(unused_metadata) >= num_needed:
                        break
                if len(unused_metadata) >= num_needed:
                    break
        except FileNotFoundError:
            print(f"Warning: No-spam source file not found: {file_path}")
        except Exception as e:
            print(f"Error reading or processing {file_path}: {e}")
            continue # Try next file if one fails
            
    random.shuffle(unused_metadata) # Shuffle to pick random metadata
    return unused_metadata

if __name__ == "__main__":
    manually_labelled_files = [
        "emotion_analysis/emotion/manually_labelled/despa_kJQP7kiw5Fk_emotion_manual_labelled_kappa76.csv",
        "emotion_analysis/emotion/manually_labelled/seeyou_RgKAFK5djSk_emotion_manual_labelled_kappa77.csv"
    ]
    no_spam_files = [
        "emotion_analysis/emotion/despa_kJQP7kiw5Fk_comments_no_spam.csv",
        "emotion_analysis/emotion/seeyou_RgKAFK5djSk_comments_no_spam.csv"
    ]

    # Determine column structure from one of the manually labelled files
    try:
        df_structure = pd.read_csv(manually_labelled_files[0])
        original_columns = df_structure.columns.tolist()
    except Exception as e:
        print(f"Error reading '{manually_labelled_files[0]}' to determine column structure: {e}")
        original_columns = [ # Fallback, ensure this matches your actual structure
            'comment_id', 'original_comment', 'comment_no_emojis', 'emojis_in_comment',
            'emoji_text_representation', 'detected_language', 'author_name', 'published_at',
            'published_at_unix', 'like_count', 'emotion_anger', 'emotion_disgust',
            'emotion_fear', 'emotion_joy', 'emotion_neutral', 'emotion_sadness',
            'emotion_surprise', 'manual_emotion_label_mb', 'manual_emotion_label_eden'
        ]
        print(f"Using fallback column structure: {original_columns}")

    used_ids = load_used_comment_ids(manually_labelled_files)
    print(f"Loaded {len(used_ids)} used comment IDs.")

    num_to_generate = 50
    available_metadata = get_unused_comment_metadata(no_spam_files, used_ids, num_to_generate, original_columns)
    print(f"Found {len(available_metadata)} unused comments with metadata.")

    generated_fear_comments_data = []

    for i in range(num_to_generate):
        generated_comment_text = generate_comment_text()
        
        row = {}
        if i < len(available_metadata):
            # Use real metadata
            meta = available_metadata[i]
            row = meta.copy() # Start with all columns from the source
            row['comment_id'] = meta.get('comment_id') # This should exist from get_unused_comment_metadata logic
            # Ensure other essential fields are present from meta or defaulted if somehow missing
            row['author_name'] = meta.get('author_name', f"AnonUserFallback_{random.randint(1000,9999)}")
            row['published_at'] = meta.get('published_at', (datetime.now(timezone.utc) - pd.Timedelta(days=random.randint(1,365))).isoformat().replace('+00:00', 'Z'))
            row['published_at_unix'] = meta.get('published_at_unix', int((datetime.now(timezone.utc) - pd.Timedelta(days=random.randint(1,365))).timestamp()))
            row['like_count'] = int(meta.get('like_count', 0))

        else:
            # Fallback: Generate somewhat realistic placeholders if not enough real metadata
            print(f"Warning: Not enough unique metadata, using generated placeholders for comment {i+1}")
            row['comment_id'] = f"feargen_fallback_{uuid.uuid4()}"
            row['author_name'] = f"User{random.randint(10000, 99999)}"
            random_past_date = datetime.now(timezone.utc) - pd.Timedelta(days=random.randint(1, 1825), hours=random.randint(0,23), minutes=random.randint(0,59)) # Up to 5 years back
            row['published_at'] = random_past_date.isoformat().replace('+00:00', 'Z')
            row['published_at_unix'] = int(random_past_date.timestamp())
            row['like_count'] = random.randint(0, 50) # Random likes for fallback

        # Overwrite specific fields for the generated fear comment
        row['original_comment'] = generated_comment_text
        row['comment_no_emojis'] = generated_comment_text 
        row['emojis_in_comment'] = ''
        row['emoji_text_representation'] = ''
        row['detected_language'] = 'en'
        
        row['emotion_anger'] = 0.0
        row['emotion_disgust'] = 0.0
        row['emotion_fear'] = 1.0
        row['emotion_joy'] = 0.0
        row['emotion_neutral'] = 0.0
        row['emotion_sadness'] = 0.0
        row['emotion_surprise'] = 0.0
        row['manual_emotion_label_mb'] = 'fear'
        row['manual_emotion_label_eden'] = 'fear'

        # Ensure all original columns are present, fill missing ones from fallback logic or default
        for col in original_columns:
            if col not in row:
                if col == 'comment_id': row[col] = f"feargen_final_fallback_{uuid.uuid4()}"
                elif col == 'author_name': row[col] = "FearCommenter"
                elif col == 'published_at': row[col] = (datetime.now(timezone.utc) - pd.Timedelta(days=random.randint(1,10))).isoformat().replace('+00:00', 'Z')
                elif col == 'published_at_unix': row[col] = int((datetime.now(timezone.utc) - pd.Timedelta(days=random.randint(1,10))).timestamp())
                elif col == 'like_count': row[col] = 0
                elif col.startswith('emotion_'): row[col] = 0.0 # should be covered
                elif col.startswith('manual_emotion_label_'): row[col] = 'fear' # should be covered
                else: row[col] = pd.NA # Or other sensible default like '' or 0 for unspecified columns

        generated_fear_comments_data.append(row)

    new_fear_df = pd.DataFrame(generated_fear_comments_data, columns=original_columns)
    # Ensure like_count is integer, others are appropriate
    new_fear_df['like_count'] = new_fear_df['like_count'].fillna(0).astype(int)
    for col in ['emotion_anger', 'emotion_disgust', 'emotion_fear', 'emotion_joy', 'emotion_neutral', 'emotion_sadness', 'emotion_surprise']:
        if col in new_fear_df.columns:
            new_fear_df[col] = new_fear_df[col].fillna(0.0).astype(float)

    output_filename = "generated_fear_comments_for_training_v2.csv"
    new_fear_df.to_csv(output_filename, index=False)

    print(f"Generated {len(new_fear_df)} new 'fear' comments and saved them to '{output_filename}'")
    print(f"Columns in the new CSV: {new_fear_df.columns.tolist()}")
    if len(available_metadata) < num_to_generate:
        print(f"Note: {num_to_generate - len(available_metadata)} comments used fallback metadata due to insufficient unique real comments.")

    # Optional: Print some generated comments to verify
    # print("First 5 generated comments with their metadata:")
    # print(new_fear_df[['comment_id', 'original_comment', 'author_name', 'published_at', 'like_count']].head())