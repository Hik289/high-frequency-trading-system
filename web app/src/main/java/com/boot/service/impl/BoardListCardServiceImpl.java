package com.boot.service.impl;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import com.boot.dao.BoardListCardDao;
import com.boot.dao.FilesDao;
import com.boot.entity.BoardListCard;
import com.boot.service.BoardListCardService;

@Service
public class BoardListCardServiceImpl implements BoardListCardService {

	@Autowired
	private BoardListCardDao boardListCardDao;
	@Autowired
	private FilesDao filesDao;
	
	@Override
	@Transactional(propagation=Propagation.REQUIRED,rollbackFor=Exception.class)
	public void save(BoardListCard listCard) {
		if(listCard!=null&&listCard.getId()!=null) 
			boardListCardDao.update(listCard);
		else
			boardListCardDao.save(listCard);
	}
	@Override
	public BoardListCard getDetailById(Integer cardId) {
		return boardListCardDao.getDetailById(cardId);
	}
	@Override
	@Transactional(propagation=Propagation.REQUIRED,rollbackFor=Exception.class)
	public void save(BoardListCard listCard, String[] fileId) {
		if(listCard!=null&&listCard.getId()!=null) {
			boardListCardDao.update(listCard);
		}else {
			boardListCardDao.save(listCard);
		}
		if(fileId!=null&&fileId.length>0) {
			//处理附件
			filesDao.deletByCard(listCard.getId());
			filesDao.addByCard(listCard.getId(),fileId);
		}
	}

}
