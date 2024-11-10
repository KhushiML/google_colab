@router.patch("/update_personality_code")
def update_personality_code(id: int, db: Session = Depends(get_db)):
    # Step 1: Fetch the sub_story record by id
    sub_story = db.query(SubStory).filter(SubStory.id == id).first()
   
    if not sub_story:
        raise HTTPException(status_code=404, detail="SubStory not found")
   
    logging.info(f"Fetched sub_story: {sub_story}")
 
    # Step 2: Use the 'personality' column from sub_story to find the corresponding per_name in the personality table
    personality = db.query(Personality).filter(Personality.per_name.ilike(sub_story.personality)).first()
 
   
    # Step 3: Extract the per_code and per_type from the personality table and if not found update code with UNKNOWN CODE  
 
    if not personality:
        last_personality = db.query(Personality).order_by(Personality.per_code.desc()).first()
        new_personality_code = (last_personality.per_code + 1) if last_personality else 1  # Start from 1 if no items exist
        personality_type = "UNKNOWN"
        new_personality = Personality(            
            per_name = "UNKNOWN",
            per_code = new_personality_code,
            )
        db.add(new_personality)
        db.commit()
        db.refresh(new_personality)
        # raise HTTPException(status_code=404, detail="UNKNOWN: Personality not found in personality table")
    else:
        personality_type = personality.per_type
        personality_code = personality.per_code
        logging.info(f"Fetched personality: {personality}")
 
 
    # Step 4: Update the PersonalityCode column in the sub_story table with the personality_code using personality type
    if personality_type=="Personality":
        sub_story.PersonalityCode = personality_code
        logging.info(f"Updated PersonalityCode to: {sub_story.PersonalityCode}")
    elif personality_type=="Guest":
        sub_story.GuestCode = personality_code
        logging.info(f"Updated GuestCode to: {sub_story.GuestCode}")
    elif personality_type=="Anchor":
        sub_story.AnchorCode = personality_code
        logging.info(f"Updated AnchorCode to: {sub_story.AnchorCode}")
    elif personality_type=="Reporter":
        sub_story.ReporterCode = personality_code
        logging.info(f"Updated ReporterCode to: {sub_story.ReporterCode}")
    else:
        sub_story.PersonalityCode = new_personality_code
        personality_code = new_personality_code
        logging.info(f"Updated Personality Codes with New Personality Codes: {new_personality_code}")
   
    db.add(sub_story)
    db.commit()    
   
    return {"message": "PersonalityCode updated successfully", "personality_code": personality_code, "personality_type": personality_type}