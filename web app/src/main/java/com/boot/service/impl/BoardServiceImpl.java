package com.boot.service.impl;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.boot.dao.BoardDao;
import com.boot.entity.Board;
import com.boot.entity.BoardList;
import com.boot.entity.BoardListCard;
import com.boot.service.BoardService;
import com.boot.vo.TokenHandler;

@Service
public class BoardServiceImpl implements BoardService{
	@Autowired
	private BoardDao boardDao;
	
	@Override
	public void addBoardSave(Board board) {
		Integer userId = TokenHandler.getBusinessId();
		board.setUserId(userId);
		boardDao.addBoardSave(board);
	}

	@Override
	public List<Board> getListByTeam(Integer teamId) {
		return boardDao.getListByTeam(teamId);
	}

	@Override
	public void listSave(BoardList list) {
		boardDao.listSave(list);
	}

	@Override
	public List<BoardList> getBoardListByBoard(Integer boardId) {
		return boardDao.getBoardListByBoard(boardId);
	}

	@Override
	public List<BoardListCard> getBoardListCard(Integer boardListId) {
		return boardDao.getBoardListCard(boardListId);
	}

	@Override
	public Board getDetailByid(Integer boardId) {
		return boardDao.getDetailByid(boardId);
	}

	@Override
	public BoardList getBoardListDetailById(Integer listId) {
		return boardDao.getBoardListDetailById(listId);
	}
}
