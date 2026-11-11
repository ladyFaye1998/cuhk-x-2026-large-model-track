"""The existing 40-class action namespace used by the reconstructed classifiers."""
ACTIONS=('Washing face','Brushing teeth','Combing hair','Undressing','Wiping hands','Getting dressed',
    'Drinking','Eating','Grabbing utensils','Pouring','Stirring','Peeling fruit','Sweeping','Mopping',
    'Washing dishes','Wiping surface','Folding clothes','Typing on a keyboard','Writing','Calling',
    'Checking the time','Reading','Turning a page','Listening to the music with headphones','Using a phone',
    'Watching TV','Playing games','Taking a selfie','Running','Squats','Jumping jacks','Stretching',
    'Standing up','Lying down','Sitting down','Lunges','Walking','Taking medicine','Massaging oneself','Checking body temperature')
LABEL_TO_ACTION={label:' '.join(action.lower().split()) for label,action in enumerate(ACTIONS)}
