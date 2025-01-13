"""
Comprehensive dataset of Indian travel-related entities.
"""

# Major cities and tourist destinations
LOCATIONS = {
    'Cities': [
        'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad',
        'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Chandigarh', 'Kochi',
        'Goa', 'Varanasi', 'Agra', 'Amritsar', 'Rishikesh', 'Udaipur',
        'Mysore', 'Darjeeling', 'Shimla', 'Manali', 'Ooty', 'Munnar',
        'Gangtok', 'Leh', 'Srinagar', 'Pushkar', 'Hampi', 'Madurai'
    ],
    
    'States': [
        'Maharashtra', 'Kerala', 'Rajasthan', 'Gujarat', 'Tamil Nadu',
        'Karnataka', 'Himachal Pradesh', 'Uttarakhand', 'Goa', 'Punjab',
        'West Bengal', 'Uttar Pradesh', 'Madhya Pradesh', 'Sikkim',
        'Arunachal Pradesh', 'Assam', 'Meghalaya', 'Ladakh'
    ],
    
    'Beach Destinations': [
        'Goa Beaches', 'Kovalam', 'Varkala', 'Puri', 'Andaman Islands',
        'Lakshadweep', 'Gokarna', 'Pondicherry', 'Mahabalipuram',
        'Diu', 'Tarkarli', 'Murudeshwara', 'Marari Beach', 'Bekal'
    ],
    
    'Hill Stations': [
        'Shimla', 'Manali', 'Darjeeling', 'Ooty', 'Munnar', 'Mussoorie',
        'Nainital', 'Kodaikanal', 'Coorg', 'Kasauli', 'Dharamshala',
        'McLeodganj', 'Lansdowne', 'Yercaud', 'Coonoor', 'Dalhousie'
    ]
}

# Historical and cultural landmarks
LANDMARKS = {
    'Historical Monuments': [
        'Taj Mahal', 'Red Fort', 'Qutub Minar', 'Hawa Mahal', 'India Gate',
        'Gateway of India', 'Victoria Memorial', 'Mysore Palace',
        'Fatehpur Sikri', 'Amber Fort', 'Mehrangarh Fort', 'Konark Sun Temple',
        'Khajuraho Temples', 'Sanchi Stupa', 'Golconda Fort', 'Charminar',
        'Brihadeeswara Temple', 'Hampi Ruins', 'Ajanta Caves', 'Ellora Caves'
    ],
    
    'Religious Sites': [
        'Golden Temple', 'Varanasi Ghats', 'Kedarnath Temple',
        'Badrinath Temple', 'Tirupati Temple', 'Meenakshi Temple',
        'Jagannath Temple', 'Shirdi Sai Baba Temple', 'Siddhivinayak Temple',
        'Basilica of Bom Jesus', 'Akshardham Temple', 'Ranakpur Jain Temple',
        'Dharamsala Monastery', 'Rumtek Monastery', 'Bodh Gaya',
        'Mahabodhi Temple', 'Amarnath Cave', 'Somnath Temple'
    ],
    
    'Modern Attractions': [
        'Statue of Unity', 'Lotus Temple', 'Science City Kolkata',
        'Ramoji Film City', 'Kingdom of Dreams', 'VGP Universal Kingdom',
        'Essel World', 'Nicco Park', 'Wonderla', 'Snow World'
    ]
}

# Cultural events and festivals
EVENTS = {
    'Religious Festivals': [
        'Diwali', 'Holi', 'Durga Puja', 'Ganesh Chaturthi', 'Navratri',
        'Onam', 'Pongal', 'Baisakhi', 'Maha Shivratri', 'Eid',
        'Christmas', 'Raksha Bandhan', 'Janmashtami', 'Karva Chauth'
    ],
    
    'Cultural Festivals': [
        'Goa Carnival', 'Pushkar Fair', 'Kumbh Mela', 'Rann Utsav',
        'Kerala Boat Race', 'Hornbill Festival', 'Ziro Music Festival',
        'Sunburn Festival', 'NH7 Weekender', 'Jaipur Literature Festival',
        'International Film Festival of India', 'Chennai Music Festival',
        'Konark Dance Festival', 'Khajuraho Dance Festival'
    ],
    
    'Food Festivals': [
        'National Street Food Festival', 'Grub Fest',
        'Goa Food and Cultural Festival', 'Rajasthan Food Festival',
        'Kerala Food Festival', 'Ahmedabad Food Festival',
        'Delhi Food Truck Festival', 'Mumbai Food Festival'
    ]
}

# Transportation and accommodation
ORGANIZATIONS = {
    'Airlines': [
        'Air India', 'IndiGo', 'SpiceJet', 'Vistara', 'Go First',
        'Air India Express', 'Alliance Air', 'TruJet'
    ],
    
    'Railway Services': [
        'Indian Railways', 'IRCTC', 'Palace on Wheels',
        'Golden Chariot', 'Maharajas Express', 'Deccan Odyssey'
    ],
    
    'Hotel Chains': [
        'Taj Hotels', 'Oberoi Hotels', 'ITC Hotels', 'Leela Palace',
        'The Park Hotels', 'Lalit Hotels', 'Vivanta', 'Ginger Hotels',
        'Fortune Hotels', 'Lemon Tree Hotels', 'Sarovar Hotels'
    ],
    
    'Travel Companies': [
        'MakeMyTrip', 'Yatra', 'Cleartrip', 'EaseMyTrip',
        'Thomas Cook India', 'Cox & Kings', 'SOTC', 'Club Mahindra'
    ]
}

# Activities and experiences
ACTIVITIES = {
    'Adventure Sports': [
        'River Rafting in Rishikesh', 'Paragliding in Bir Billing',
        'Skiing in Gulmarg', 'Scuba Diving in Andamans',
        'Trekking in Ladakh', 'Rock Climbing in Hampi',
        'Bungee Jumping in Rishikesh', 'Zip Lining in Neemrana'
    ],
    
    'Wellness': [
        'Yoga in Rishikesh', 'Ayurveda in Kerala',
        'Meditation in Dharamsala', 'Spa Treatments in Udaipur',
        'Naturopathy in Bangalore', 'Wellness Retreat in Dehradun'
    ],
    
    'Cultural Experiences': [
        'Cooking Classes in Jaipur', 'Block Printing in Bagru',
        'Pottery Making in Delhi', 'Classical Dance in Chennai',
        'Silk Weaving in Varanasi', 'Kathakali Learning in Kerala'
    ]
}

def get_all_locations():
    """Get all location entities."""
    locations = []
    for category in LOCATIONS.values():
        locations.extend(category)
    return list(set(locations))

def get_all_landmarks():
    """Get all landmark entities."""
    landmarks = []
    for category in LANDMARKS.values():
        landmarks.extend(category)
    return list(set(landmarks))

def get_all_events():
    """Get all event entities."""
    events = []
    for category in EVENTS.values():
        events.extend(category)
    return list(set(events))

def get_all_organizations():
    """Get all organization entities."""
    organizations = []
    for category in ORGANIZATIONS.values():
        organizations.extend(category)
    return list(set(organizations))

def get_all_activities():
    """Get all activity descriptions."""
    activities = []
    for category in ACTIVITIES.values():
        activities.extend(category)
    return list(set(activities))
