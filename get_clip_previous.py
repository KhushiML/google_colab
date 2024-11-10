@router.get("/get_clip_url_for_personality/", response_model=SubStoryReadModel2)
async def get_clip_url_for_personality(db: Session = Depends(get_db)):
    #sub_story = db.query(SubStory).filter(SubStory.news_type == None).first()
    sub_story = db.query(SubStory).filter(
        SubStory.personality_status == 0
    ).first()
    print(sub_story)
    if not sub_story:
        raise HTTPException(status_code=404, detail="No SubStory found")
    return sub_story