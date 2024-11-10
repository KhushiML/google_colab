@router.get("/get_clip_url_for_personality/", response_model=SubStoryReadModel2)
async def get_clip_url_for_personality(db: Session = Depends(get_db)):
    sub_story = db.query(SubStory).filter(
        SubStory.personality_status == 0,
        SubStory.duration_seconds > 1
    ).first()
    
    print(sub_story)
    if not sub_story:
        print("No SubStory found with personality_status == 0 and duration_seconds > 1")
        raise HTTPException(status_code=404, detail="No SubStory found")
    
    return sub_story