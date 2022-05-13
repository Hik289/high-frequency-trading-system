package com.boot.service;

import com.boot.entity.BoardListCard;

public interface BoardListCardService {

	void save(BoardListCard listCard);

	BoardListCard getDetailById(Integer cardId);

	void save(BoardListCard listCard, String[] fileId);
	
}
